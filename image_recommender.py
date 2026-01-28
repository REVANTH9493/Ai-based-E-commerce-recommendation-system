import streamlit as st
import requests
from io import BytesIO
import os
import base64
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import InferenceClient

# Load env variables explicitly
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

# Force-check token existence
if not HF_TOKEN:
    st.error("⚠️ Missing Hugging Face Token! Please set 'HF_TOKEN' in your .env file or Streamlit secrets.")
    st.stop()

# Initialize Client
# We use the default inference provider (Hugging Face Inference API) unless specified otherwise
client = InferenceClient(token=HF_TOKEN)
MODEL_ID = "sentence-transformers/clip-ViT-B-32"

# -----------------------------
# Image Loading Helper
# -----------------------------
@st.cache_data
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=3)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except:
        return None

# -----------------------------
# Feature Extraction (Cached)
# -----------------------------

@st.cache_data
def get_dataset_features(data):
    """
    Loads pre-computed embeddings from file since we cannot compute them live on Cloud.
    """
    # Try loading from local file (expected to be in repo)
    try:
        if os.path.exists("text_embeddings_cache.npy"):
            all_features = np.load("text_embeddings_cache.npy")
            # We assume indices align with data.index if data hasn't changed drastically.
            if len(all_features) == len(data):
                return all_features, data.index.tolist()
            else:
                print("Warning: Content mismatch between cache and live data.")
                min_len = min(len(all_features), len(data))
                return all_features[:min_len], data.index.tolist()[:min_len]
        return np.array([]), []
    except Exception as e:
        st.error(f"Error loading cache: {e}")
        return np.array([]), []


def get_uploaded_image_embedding(uploaded_image_obj):
    """
    Sends uploaded image to HF API for feature extraction using InferenceClient.
    Handles both Streamlit UploadedFile and PIL Image.
    """
    try:
        # 1. Prepare Image Object for Client
        # InferenceClient accepts path, URL, or bytes/file-like object
        # If it's a PIL Image, convert to bytes first
        img_for_api = uploaded_image_obj
        
        if isinstance(uploaded_image_obj, Image.Image):
             img_byte_arr = BytesIO()
             uploaded_image_obj.save(img_byte_arr, format='PNG')
             img_for_api = img_byte_arr.getvalue()
        
        # 2. Call Feature Extraction
        # This automatically handles the API call
        # Returns a numpy array directly usually
        response = client.feature_extraction(img_for_api, model=MODEL_ID)
        
        return response
    except Exception as e:
        st.error(f"Error fetching embedding via InferenceClient: {e}")
        return None

def recommend_by_image(uploaded_image, data=None, top_n=5):
    """
    Main function to be called from the Streamlit App.
    Returns a dataframe of recommended products.
    """
    # 1. Load Data if not provided
    if data is None:
        from firebase_utils import get_data_from_firebase
        from preprocess_data import process_data
        
        raw_data = get_data_from_firebase()
        if raw_data is None or raw_data.empty:
            st.error("No data available for recommendations.")
            return pd.DataFrame()
        data = process_data(raw_data)
    
    # 2. Get Dataset Features (Cached)
    with st.spinner("Indexing product catalog..."):
        dataset_features, valid_indices = get_dataset_features(data)
    
    if len(dataset_features) == 0:
        st.error("Could not extract features from product catalog.")
        return pd.DataFrame()

    # 3. Process Query Image (via API)
    with st.spinner("Analyzing image..."):
        query_embedding_np = get_uploaded_image_embedding(uploaded_image)
        
    if query_embedding_np is None:
        return pd.DataFrame()
        
    try:
        # Result from InferenceClient is often already an ndarray
        # Ensure it's the right shape
        if isinstance(query_embedding_np, list):
            query_embedding_np = np.array(query_embedding_np)
            
        # If wrapped in extra dimensions (e.g. batch), flatten it
        if query_embedding_np.ndim == 2:
            query_embedding_np = query_embedding_np.mean(axis=0) 
        elif query_embedding_np.ndim > 2:
             query_embedding_np = query_embedding_np.flatten() 
            
        # Normalize
        norm = np.linalg.norm(query_embedding_np)
        if norm > 0:
            query_embedding_np = query_embedding_np / norm
            
        # 4. Compute Similarity
        similarities = (dataset_features @ query_embedding_np.T)
        if similarities.ndim > 1:
            similarities = similarities.squeeze()
            
        # 5. Get Top N
        top_indices_local = similarities.argsort()[::-1][:top_n]
        top_df_indices = [valid_indices[i] for i in top_indices_local]
        safe_indices = [i for i in top_df_indices if i in data.index]
        
        recommended_df = data.loc[safe_indices].copy()
        return recommended_df
        
    except Exception as e:
        st.error(f"Processing Error: {e}")
        return pd.DataFrame()


# Placeholder function can be removed or kept as empty
def get_text_embeddings(text):
    return None