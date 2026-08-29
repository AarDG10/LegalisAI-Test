# LegalisAI: Real Estate Legal Case Assistant
A specialized legal assistant leveraging the power of the **InLegalBERT** model to respond to user queries related to real estate cases (Info. Retrieval + Deep Learning Model) based solution.

---

## 🔧 **Principal Architecture** (Model + Training)
1. **Model:** [InLegalBERT Model](https://huggingface.co/law-ai/InLegalBERT)
2. **Training Data:** Real Estate Legal Cases Dataset

## 🧱 **Stack**
- `legalis_api/` — FastAPI backend: owns the models, embeddings, and retrieval logic.
- `app.py` — Streamlit UI, calls the FastAPI backend over HTTP. Run the API first, then this.

## ▶️ **Running it**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r legalis_api/requirements.txt
pip install googletrans==4.0.0-rc1 --no-deps  # see note in requirements.txt

# terminal 1
cd legalis_api && uvicorn main:app --reload

# terminal 2
streamlit run app.py
```
Requires `legalis_model/`, `faq_model/`, and `Data/` (gitignored) present at the repo root — see `.gitignore`.

---

## 🖥️ **I/O Description**

- **Input:** User-provided case description
- **Output:**
  - Relevant sections of law
  - Relevancy score of sections
  - Strong and weak points associated with the case
- **Added Features**
  - Language Compatibility (Eng/Hindi/Marathi)

---

## ⚠️ Disclaimer
This tool is informational only and does not constitute legal advice. See
`NOTICE.md` for the full disclaimer, licensing (MIT), and model attribution.

## 📄 Note
The curated case dataset is not published in this repository — see
`NOTICE.md`.
