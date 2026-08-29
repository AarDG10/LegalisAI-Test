import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json
import os

# Load InLegalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")


# Save the model and tokenizer to the "faq_model" folder
def save_model(model, tokenizer, model_path="faq_model"):
    os.makedirs(model_path, exist_ok=True)
    # Save the model state_dict
    torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))

    # Save the tokenizer
    tokenizer.save_pretrained(model_path)

    # Manually create and save a config.json if it does not exist
    config = AutoConfig.from_pretrained("law-ai/InLegalBERT")
    config.save_pretrained(model_path)  # Save config to the same directory

    print(f"Model, tokenizer, and config saved at {model_path}")


# Save the model
save_model(model, tokenizer)


# Function to get embeddings for text
def get_embeddings(texts):
    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()  # Return as NumPy array for FAISS


# Load data from JSONL file with explicit encoding
def load_faq_data(file_path):
    faq_data = []
    with open(file_path, "r", encoding="utf-8") as f:  # Specify encoding as 'utf-8'
        for line in f:
            faq_data.append(json.loads(line))
    return faq_data


# Load FAQ data from a JSONL file
faq_data = load_faq_data("QandA.jsonl")

# Step 1: Get embeddings for FAQ prompts
faq_prompts = [faq["prompt"] for faq in faq_data]
faq_embeddings = get_embeddings(faq_prompts)

# Step 2: Initialize FAISS index
dim = faq_embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean distance)
index.add(faq_embeddings)  # Add FAQ embeddings to FAISS index


# Function to find the most similar FAQ for a query
def find_similar_faq(query):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(
        query_embedding, k=1
    )  # Search for the top 1 most similar FAQ
    return faq_data[indices[0][0]], distances[0][0]
