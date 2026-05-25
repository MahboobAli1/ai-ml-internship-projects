import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = r"C:\Users\Mahboob Ali\Developer Hb\Phase 2\news-classifier-app"

# 🔥 USE EXPLICIT TOKENIZER (NO AutoTokenizer)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels = ["World", "Sports", "Business", "Sci/Tech"]

st.title("📰 News Classifier")

text = st.text_area("Enter News Headline")

if st.button("Predict"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    st.success(labels[pred])