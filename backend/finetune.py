import pandas as pd
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from datasets import Dataset

#Connecting to database
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["interactly"]
collection = db["candidates sample"]

data = list(collection.find())
df = pd.DataFrame(data)
model = SentenceTransformer("all-MiniLM-L6-v2")

#Preprocessing and creating embeddings
candidate_embeddings = []
for _, row in df.iterrows():
    profile_text = f"{row['Name']} {row['Job Skills']} {row['Experience']} {row['Projects']} {row['Comments']}"
    embedding = model.encode(profile_text)
    candidate_embeddings.append(embedding)

candidate_embeddings = np.array(candidate_embeddings)
dimension = candidate_embeddings.shape[1]

#Indexing embeddings using Faiss
index = faiss.IndexFlatL2(dimension)
index.add(candidate_embeddings)
print("Data preprocessing and indexing completed.")

embeddings_file = "embeddings.pkl"
metadata_file = "metadata.pkl"

with open(embeddings_file, "wb") as f:
    pickle.dump(candidate_embeddings, f)

with open(metadata_file, "wb") as f:
    pickle.dump(df.to_dict(), f)

print("Embeddings and metadata saved.")
resume_df = pd.read_csv("C:/myprojects/interactly-rag/backend/Resume.csv")
resume_df = resume_df.sample(100)  

#Finetuning
dataset = Dataset.from_pandas(resume_df[["Resume_str"]])

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

model.resize_token_embeddings(len(tokenizer))


def tokenize_function(examples):
    tokens = tokenizer(
        examples["Resume_str"], truncation=True, padding="max_length", max_length=128
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
print("Fine-tuning completed.")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
