# RegDoc: AI Powered Regulatory Document Classifier

Hitachi Digital Services | Texas A&M Datathon 2025

---

## Overview

RegDoc is an AI powered document intelligence system designed to automatically analyze and classify regulatory and business documents into four categories: Public, Confidential, Highly Sensitive, and Unsafe.

It supports both multi-page PDFs and image-based documents, combining deterministic policy checks with advanced language model reasoning. The system was built as part of the Hitachi Digital Services x Texas A&M Datathon 2025, with the goal of enabling fast, explainable, and audit-ready compliance classification.

- Public  
- Confidential  
- Highly Sensitive  
- Unsafe  

The system combines:

- Text and image preprocessing  
- Dynamic prompt tree generation from a configurable prompt library  
- Dual model validation using two LLMs  
- Human in the loop (HITL) review and overrides  
- Citation based evidence for each classification decision  

---

## System Architecture

### High Level Flow

```text
User Upload (PDF / Image)
        |
        v
Ingestion Layer (OCR + Text)
        |
        v
Heuristics and Safety Layer
        |
        v
LLM Classification Engine
        |
        v
Policy Rules and Overrides
        |
        v
Streamlit HITL Interface + Audit Log
```
---

## Category Logic

| Category | Example Documents | Trigger Logic |
|-----------|------------------|----------------|
| **Public** | Brochures, press releases, marketing content | Default if no PII or restricted terms detected |
| **Confidential** | Internal memos, project proposals, technical reports | Contains “internal use”, “restricted circulation”, or equipment terms |
| **Highly Sensitive** | Employment forms, application data with SSNs or credit cards | PII detection (SSN, account number, or address) |
| **Unsafe** | Explicit, violent, or illegal content | Matches unsafe keyword list (CSAM, self harm, violence) |

`kid_safe` is marked **False** if strong profanity is detected, even if the document is not unsafe in other respects.

### Step-by-Step Pipeline

```text
1. Pre-processing
   - Extract text from PDFs using pdfplumber
   - OCR images with pytesseract
   - Count pages, images, and check legibility

2. Heuristic Detection
   - Identify PII (email, phone, SSN, credit card)
   - Flag profanity and unsafe keywords
   - Detect aircraft or serial numbers for equipment sensitivity

3. Prompt Construction
   - Selects prompt sets from /prompts (public, sensitive, unsafe)
   - Builds a context-aware system prompt for the LLM

4. LLM Inference
   - Runs the primary model (LLaMA 3.1 8B Instruct)
   - If confidence < 0.6 → runs validator (LLaMA 3.1 70B Instruct)
   - Merges results and citations if disagreement occurs

5. Policy Enforcement
   - Internal or restricted wording → Confidential  
   - SSN or PII → Highly Sensitive  
   - Unsafe keywords → Unsafe  
   - Sensitive equipment → Confidential  

6. Human-in-the-Loop Review
   - Reviewer validates, overrides, or approves classification  
   - Feedback stored in `history.json` for continuous improvement
```
---
## Confidence Calibration and Human-in-the-Loop (HITL)

To minimize manual effort while maintaining accuracy, RegDoc uses confidence thresholds and reviewer feedback loops.

### Confidence Thresholds

| Stage | Model | Confidence Range | Action |
|--------|--------|------------------|--------|
| Primary | LLaMA 3.1 8B Instruct | ≥ 0.6 | Accept classification automatically |
| Validation | LLaMA 3.1 70B Instruct | < 0.6 | Re-evaluate and merge reasoning |
| Review | Human Reviewer | N/A | Optional override and comment |

### HITL Workflow

1. AI generates reasoning and citations for transparency.  
2. Reviewer can confirm or change the classification.  
3. Overrides and comments are saved in `history.json`.  
4. Future prompts can be tuned using this feedback.

### Benefits

- **Reduced Manual Load:** Only low-confidence cases need review.  
- **Explainable Decisions:** Each classification includes reasoning and citations.  
- **Continuous Improvement:** Reviewer feedback forms a retraining dataset.  
- **Audit Ready:** Complete trace of AI + Human actions for compliance.

## Tech Stack

| Layer | Technology |
|--------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.10+, pdfplumber, Pillow, pytesseract |
| **AI Models** | Meta LLaMA 3.1 8B and 70B Instruct via OpenRouter |
| **Environment Config** | `.env` file with `OPENROUTER_API_KEY` |
| **Storage** | JSON audit log (`history.json`) |

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-project-folder>
```
### 2. Install and Create a Virtual Environment

#### Install `virtualenv`

### 3. Install Dependencies

####Install all required dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

#### Create a `.env` File

#### In the project root folder, create a file named `.env` and paste your OpenRouter API key inside it in the following format:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 5. Run the Application

####Launch the Streamlit app:

```bash
streamlit run app.py
```

