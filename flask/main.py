from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from typing import Optional
import torch
from transformers import BertTokenizer, BertModel
import re
import json
import spacy
import nltk
from datetime import datetime, timedelta
import os
from supabase import create_client, Client
import time
import threading
import dotenv
from contextlib import asynccontextmanager

dotenv.load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up the application...")
    start_periodic_task()
    yield
    print("🛑 Shutting down the application...")

app = FastAPI(lifespan=lifespan)

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))


# Initialize Supabase client
print("🔌 Initializing Supabase connection...")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    print("❌ Missing Supabase credentials! Please check your .env file")
else:
    supabase: Client = create_client(supabase_url, supabase_key)
    print("✅ Supabase client initialized")

# Load contractions
with open("contradictions.json") as f:
    contractions = json.load(f)

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="v3.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    interpreter = None
    print(f"❌ Failed to load TFLite model: {e}")

# Request schema
class PredictionRequest(BaseModel):
    text: str
    model_type: Optional[str] = "tflite"  # for future compatibility

def periodic_database_check():
    while True:
        print("🔄 Checking the database...")
        try:
            check_database()
        except Exception as e:
            print(f"❌ Error checking database: {e}")
        time.sleep(3600)  # Sleep for one hour (3600 seconds)

def start_periodic_task():
    print("Starting periodic task...")
    thread = threading.Thread(target=periodic_database_check)
    thread.daemon = True  # Allow the main Flask app to exit even if the thread is running
    thread.start()

def check_database():
    # get all records from the reports table
    print("📊 Fetching records from reports table...")
    response = supabase.table("reports").select("*").execute()
    records = response.data
    print(f"✅ Found {len(records)} records in the database")


@app.get("/")
async def root():
    return {"message": "API is live!"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        if interpreter is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        cleaned = clean_data(request.text)
        tokens = word_tokenize(cleaned)
        tokens = remove_stopwords(tokens)
        lemmatized = lemmatize(tokens)
        embedded = get_bert_embedding(" ".join(lemmatized))  # shape: (1, 768)

        # Set tensor & invoke
        interpreter.set_tensor(input_details[0]['index'], embedded.astype(np.float32))

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(output_data))
        # - Sadness - 0
        # - Joy - 1
        # - Love - 2
        # - Anger - 3
        # - Fear - 4
        # - Surprise - 5
        
        # get the class name
        predicted_class_name = {
            0: "Sadness",
            1: "Joy",
            2: "Love",
            3: "Anger",
            4: "Fear",
            5: "Surprise"
        }[predicted_class]
        
        return {
            "prediction": predicted_class_name,
            "probabilities": output_data.tolist(),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Preprocessing functions ---

def clean_data(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return replace_contractions(text)

def replace_contractions(text):
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def word_tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in stop_words]

def lemmatize(tokens):
    return [token.lemma_ for token in nlp(" ".join(tokens))]

def get_bert_embedding(text: str):
    # BERT Embedding
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = bert_model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()  # (1, 768)

    # Extra feature: sentence length
    sentence_length = np.array([[len(text.split())]])  # shape (1, 1)

    # Concatenate to form (1, 769)
    full_vector = np.concatenate([embedding, sentence_length], axis=1)

    return full_vector

if __name__ == '__main__':
    app.run(debug=True)