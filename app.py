import concurrent.futures
import streamlit as st
from backend.ingestion import process_file
from backend.classification import classify_document
from backend.storage import save_result, load_history

st.set_page_config(page_title="RegDoc Classifier", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Analyze", "History & Audit"])

st.sidebar.markdown("---")
st.sidebar.caption("Hitachi DS Datathon • AI-Powered Regulatory Classifier")

# ----------------------------------------------------------------------
# PAGE 1: Upload & Analyze (multi-file, parallel, with SUMMARY + HITL)
# ----------------------------------------------------------------------
if page == "Upload & Analyze":
    st.title("Upload & Analyze Documents")
    st.caption(
        "Upload one or more documents. The app will classify them, show page-level summaries, "
        "and let you approve or override each result."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF or image files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    run_clicked = st.button(
        "Run Analysis",
        type="primary",
        disabled=not uploaded_files,
    )

    if run_clicked and uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected.")
        progress = st.progress(0.0)
        status_placeholder = st.empty()
        results = []

        def process_single(uploaded_file):
            """Ingest + classify a single file (used in threads)."""
            doc_info = process_file(uploaded_file)
            ai_result = classify_document(doc_info)
            return {
                "filename": uploaded_file.name,
                "doc_info": doc_info,
                "ai_result": ai_result,
            }

        total = len(uploaded_files)
        done = 0

        # Run all docs concurrently (good for I/O + network-bound LLM calls)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, total)
        ) as executor:
            future_to_file = {
                executor.submit(process_single, f): f for f in uploaded_files
            }
            for future in concurrent.futures.as_completed(future_to_file):
                f = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    done += 1
                    progress.progress(done / total)
                    status_placeholder.write(f"Finished: {f.name}")
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")

        progress.empty()
        status_placeholder.empty()
        st.success("All files processed.")

        # ---- SHOW RESULTS + SUMMARY + HITL FOR EACH DOC ----
        for item in results:
            filename = item["filename"]
            doc_info = item["doc_info"]
            ai_result = item["ai_result"]

            st.markdown("---")
            st.subheader(f"{filename}")

            # Stats / metrics
            # Stats / metrics (includes image count)
            col_top1, col_top2, col_top3, col_top4, col_top5 = st.columns(5)
            col_top1.metric("AI Category", ai_result.get("category", "—"))
            col_top2.metric("Unsafe", "Yes" if ai_result.get("unsafe") else "No")
            col_top3.metric("Kid-safe", "Yes" if ai_result.get("kid_safe") else "No")
            col_top4.metric(
                "Confidence",
                f"{ai_result.get('confidence', 0.0) * 100:.0f}%"
            )
            col_top5.metric("Image Count", doc_info.get("num_images", 0))

            # AI reasoning
            st.markdown("**AI Reasoning**")
            st.write(ai_result.get("reasoning", "No reasoning provided."))

            # Citations
            citations = ai_result.get("citations") or []
            if citations:
                st.markdown("**Citations**")
                st.table(citations)

            # 🔹 SUMMARY: page-level text preview (this is what was “missing”)
            pages = doc_info.get("pages", [])
            with st.expander("Document summary", expanded=False):
                if not pages:
                    st.write("No text was extracted from this document.")
                else:
                    for p in pages:
                        page_num = p.get("page_num") or p.get("page") or "?"
                        text = (p.get("text") or "").strip()
                        if len(text) > 600:
                            text_display = text[:600] + "..."
                        else:
                            text_display = text
                        st.markdown(f"**Page {page_num}**")
                        if text_display:
                            st.write(text_display)
                        else:
                            st.write("_No text on this page._")

            # 🧠 Human-in-the-loop review
            st.markdown("### Human Review")
            override_choice = st.selectbox(
                "Override AI category (optional)",
                ["No override", "Public", "Confidential", "Highly Sensitive", "Unsafe"],
                key=f"override_{filename}",
            )

            reviewer_comment = st.text_area(
                "Reviewer comment",
                key=f"comment_{filename}",
                placeholder="Explain why you approved or changed the AI decision...",
            )

            if st.button(f"Save"):
                final_category = (
                    ai_result.get("category", "Public")
                    if override_choice == "No override"
                    else override_choice
                )
                try:
                    save_result(
                        filename=filename,
                        doc_info=doc_info,
                        ai_result=ai_result,
                        final_category=final_category,
                        reviewer_comment=reviewer_comment.strip(),
                    )
                    st.success(f"Review for '{filename}' saved to history.json.")
                except Exception as e:
                    st.error(f"Could not save review for {filename}: {e}")

# ----------------------------------------------------------------------
# PAGE 2: History & Audit
# ----------------------------------------------------------------------
elif page == "History & Audit":
    st.title("History & Audit Trail")

    history = load_history()
    if not history:
        st.info(
            "No documents processed yet. Go to 'Upload & Analyze', run a document, "
            "and click 'Save Review'."
        )
    else:
        import pandas as pd

        df = pd.DataFrame(history)

        st.subheader("Processed Documents")
        df_sorted = df.sort_values("timestamp", ascending=False)

        show_cols = [
            c for c in [
                "timestamp",
                "filename",
                "ai_category",
                "final_category",
                "unsafe",
                "kid_safe",
                "confidence",
                "reviewer_comment",
            ]
            if c in df_sorted.columns
        ]

        st.dataframe(df_sorted[show_cols], use_container_width=True)

        filenames = df_sorted["filename"].unique().tolist()
        selected = st.selectbox("Select a document to inspect", filenames)

        doc_rows = df_sorted[df_sorted["filename"] == selected]
        latest = doc_rows.iloc[0]

        st.markdown(f"### Latest review for: `{selected}`")
        st.write(f"**AI Category:** {latest.get('ai_category')}")
        st.write(f"**Final Category:** {latest.get('final_category')}")
        st.write(f"**Unsafe:** {latest.get('unsafe')}")
        st.write(f"**Kid-safe:** {latest.get('kid_safe')}")
        conf_val = latest.get("confidence")
        if conf_val is not None:
            try:
                st.write(f"**Confidence:** {float(conf_val):.2f}")
            except Exception:
                st.write(f"**Confidence:** {conf_val}")
        st.write(f"**Reviewer comment:** {latest.get('reviewer_comment') or '—'}")
        st.write(f"**Timestamp (UTC):** {latest.get('timestamp')}")
