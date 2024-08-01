import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load models and tokenizers
model_classification = BertForSequenceClassification.from_pretrained('./query_classification_model')
tokenizer_classification = BertTokenizer.from_pretrained('./query_classification_model')

model_sub_classification = BertForSequenceClassification.from_pretrained('./sub_classification_model')
tokenizer_sub_classification = BertTokenizer.from_pretrained('./sub_classification_model')

def classify_query(query):
    inputs = tokenizer_classification(query, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model_classification(**inputs)
    logits = outputs.logits
    classification = torch.argmax(logits, dim=1).item()
    return classification

def sub_classify_query(query):
    inputs = tokenizer_sub_classification(query, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model_sub_classification(**inputs)
    logits = outputs.logits
    sub_classification = torch.argmax(logits, dim=1).item()
    return sub_classification

def route_query(query):
    classification = classify_query(query)
    if classification == 0:
        return "text_generation"
    else:
        sub_classification = sub_classify_query(query)
        return f"document_retrieval_collection_{sub_classification}"

# Implementing the RAG Framework
def process_query(query):
    route = route_query(query)
    if route == "text_generation":
        response = generate_text_response(query)
    else:
        collection = route.split('_')[-1]
        documents = retrieve_documents(query, collection)
        response = generate_document_response(documents)
    return response

def generate_text_response(query):
    return "Generated text response."

def retrieve_documents(query, collection):
    return ["Document 1", "Document 2"]

def generate_document_response(documents):
    return "Generated document response based on retrieved documents."

st.title("Query Classification and Routing")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        response = process_query(query)
        st.write(f"Response: {response}")
    else:
        st.write("Please enter a query.")
