# LegalisAI: Real Estate Legal Case Assistant
A retrieval-based legal assistant that surfaces similar past MahaRERA case outcomes,
strong/weak points, and relevant RERA sections for a user's real estate dispute
(Information Retrieval, not fine-tuned/trained on this data — see `NOTICE.md`).

---

## 🔧 **Principal Architecture**
1. **Embedding model:** [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
   — pretrained sentence-embedding model, used as-is (no fine-tuning). See
   `eval/` for retrieval-quality measurements.
2. **Corpus:** curated MahaRERA case summaries + RERA FAQ pairs (not published — see `NOTICE.md`).

## 🧱 **Stack**
- `legalis_api/` — FastAPI backend: embedding model, precomputed retrieval index, retrieval logic.
- `app.py` — Streamlit UI, calls the FastAPI backend over HTTP. Run the API first, then this.
- `eval/` — retrieval-quality eval harness (hand-labeled queries + Hit@k/MRR scoring). Run
  `python eval/run_eval.py` before and after any retrieval change to measure impact.

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
Requires `Data/` (gitignored) present at the repo root — see `.gitignore`. The embedding
model downloads automatically from Hugging Face on first run.

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
