import json
import jsonlines
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to your models
legalis_model_path = "../legalis_model"
faq_model_path = "../faq_model"

# Load tokenizers and models for both Legalis and FAQ
tokenizer_legalis = AutoTokenizer.from_pretrained(legalis_model_path)
model_legalis = AutoModel.from_pretrained(legalis_model_path)
model_legalis.eval()

tokenizer_faq = AutoTokenizer.from_pretrained(faq_model_path)
model_faq = AutoModel.from_pretrained(faq_model_path)
model_faq.eval()

# Load Legalis Data from JSON
try:
    with open("../Data/finalcases.json", "r") as f:
        cases_data = json.load(f)  # This assumes the JSON is an array of case objects.
except FileNotFoundError:
    logger.error("Legalis data file not found.")
    cases_data = []

# Load FAQ Data from JSONL
faq_data = []
try:
    with jsonlines.open("../Data/QandA.jsonl") as reader:
        for obj in reader:
            faq_data.append(obj)
except FileNotFoundError:
    logger.error("FAQ data file not found.")


# Pydantic model for the request body
class TextRequest(BaseModel):
    text: str
    model_choice: str = Field(..., pattern="^(legalis|faq)$", example="legalis")


# Encode a batch of texts in a single forward pass.
def encode_texts(texts, tokenizer, model):
    if not texts:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


# --- Precompute embeddings once at startup instead of on every request ---

case_vectors = encode_texts(
    [case["case_description"] for case in cases_data], tokenizer_legalis, model_legalis
)

# Flatten all case sections into one batch, embed once, then split back per case.
_section_counts = [len(case["sections"]) for case in cases_data]
_all_section_texts = [
    section["section_description"]
    for case in cases_data
    for section in case["sections"]
]
_all_section_vectors = encode_texts(
    _all_section_texts, tokenizer_legalis, model_legalis
)

section_vectors_per_case = []
_offset = 0
for count in _section_counts:
    section_vectors_per_case.append(_all_section_vectors[_offset : _offset + count])
    _offset += count

faq_vectors = encode_texts(
    [faq["prompt"] for faq in faq_data], tokenizer_faq, model_faq
)

logger.info(
    f"Precomputed embeddings for {len(cases_data)} cases and {len(faq_data)} FAQs."
)


# Function to find relevant cases (Legalis) with most similar sections
def find_relevant_cases(user_input, num_results=5):
    query_vector = encode_texts([user_input], tokenizer_legalis, model_legalis)
    similarities = cosine_similarity(query_vector, case_vectors).flatten()

    top_indices = np.argsort(similarities)[-num_results:][::-1]

    results = []
    for index in top_indices:
        case = cases_data[index]
        section_vectors = section_vectors_per_case[index]

        if len(section_vectors) > 0:
            section_similarities = cosine_similarity(
                query_vector, section_vectors
            ).flatten()
            top_section_indices = np.argsort(section_similarities)[-3:][::-1]
            top_sections = [case["sections"][i] for i in top_section_indices]
        else:
            top_sections = []

        results.append(
            {
                "case_id": case["case_id"],
                "case_title": case["case_title"],
                "case_link": case["case_link"],
                "similarity_score": float(similarities[index]),
                "sections": top_sections,
                "strong_points": case["strong_points"],
                "weak_points": case["weak_points"],
            }
        )

    return results


# Function to find relevant FAQs (FAQ Model)
def find_relevant_faq(query, num_results=5):
    query_vector = encode_texts([query], tokenizer_faq, model_faq)
    similarities = cosine_similarity(query_vector, faq_vectors).flatten()

    top_indices = np.argsort(similarities)[-num_results:][::-1]

    results = []
    for index in top_indices:
        faq = faq_data[index]
        results.append(
            {
                "faq_prompt": faq["prompt"],
                "faq_completion": faq["completion"],
                "similarity_score": float(similarities[index]),
            }
        )

    return results


# Root endpoint for checking if the API is up
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Legalis AI API!"}


# Prediction endpoint (POST)
@app.post("/predict/")
async def predict(request: TextRequest):
    try:
        # Input validation
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        if request.model_choice == "legalis":
            if not cases_data:
                raise HTTPException(status_code=404, detail="No legal cases available.")
            result = find_relevant_cases(request.text)
            if result:
                return {"model": "Legalis", "results": result}
            else:
                raise HTTPException(status_code=404, detail="No relevant cases found.")

        elif request.model_choice == "faq":
            if not faq_data:
                raise HTTPException(status_code=404, detail="No FAQs available.")
            result = find_relevant_faq(request.text)
            if result:
                return {"model": "FAQ", "results": result}
            else:
                raise HTTPException(status_code=404, detail="No relevant FAQs found.")

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model_choice. Please choose either 'legalis' or 'faq'.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Testing Locally Command (for reference)
# curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d "{\"text\": \"What is the procedure for property registration?\", \"model_choice\": \"legalis\"}"
# curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d "{\"text\": \"How do I register a property in Maharashtra?\", \"model_choice\": \"faq\"}"
