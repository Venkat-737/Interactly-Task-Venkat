import pandas as pd
import numpy as np
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify

load_dotenv()
app = Flask(__name__)
CORS(app)
client = MongoClient(os.getenv("MONGO_URI"))
db = client["interactly"]
collection = db["candidates sample"]

data = list(collection.find())
df = pd.DataFrame(data)

# Loading saved embeddings and metadata
with open("embeddings.pkl", "rb") as f:
    candidate_embeddings = pickle.load(f)

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


df = pd.DataFrame(metadata)

dimension = candidate_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(candidate_embeddings)
print("FAISS index initialized with embeddings.")

model_name = "./fine_tuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

query_encoder = SentenceTransformer("all-MiniLM-L6-v2")


# Implementing RAG to combine indexed embeddings with finetuned LLM
def retrieve_and_generate(query, top_k=5):

    query_embedding = query_encoder.encode(query, convert_to_tensor=True)
    distances, indices = index.search(np.array([query_embedding]), k=top_k)

    responses = []
    for idx in indices[0]:
        candidate = df.iloc[idx]
        profile_text = (
            f"Name: {candidate['Name']},\n"
            f"Job Skills: {candidate['Job Skills']},\n"
            f"Experience: {candidate['Experience']},\n"
            f"Projects: {candidate['Projects']},\n"
            f"Comments: {candidate['Comments']}"
        )
        response = generator(
            profile_text, max_length=50, num_return_sequences=1, truncation=True
        )
        responses.append(response[0]["generated_text"])

    return responses


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        job_description = data.get("job_description")
        if not job_description:
            return jsonify({"error": "Job description is needed"}), 400

        responses = retrieve_and_generate(job_description)
        return jsonify({"responses": responses})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
