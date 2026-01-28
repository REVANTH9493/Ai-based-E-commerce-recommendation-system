import streamlit as st
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from firebase_utils import get_data_from_firebase
from preprocess_data import process_data

# -----------------------------
# Device setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load CLIP model (Cached)
# -----------------------------
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

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
def extract_features_safe(outputs):
    """
    Safely extracts a tensor from various Hugging Face model outputs.
    """
    # 1. If it's already a tensor, return it
    if isinstance(outputs, torch.Tensor):
        return outputs
        
    # 2. Try common attribute names for embeddings/pooled output
    if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
        return outputs.text_embeds
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
        # Fallback: Mean pooling or CLS token if pooler_output is missing
        return outputs.last_hidden_state.mean(dim=1)
        
    # 3. Try tuple indexing (outputs[1] is often pooled, [0] is last_hidden)
    if isinstance(outputs, (tuple, list)):
        if len(outputs) > 1:
            return outputs[1] # Potential pooler_output
        elif len(outputs) > 0:
            return outputs[0].mean(dim=1) if outputs[0].dim() == 3 else outputs[0]
            
    # 4. If all else fails, return outputs (will likely crash downstream but we tried)
    return outputs

@st.cache_data
def get_dataset_features(data):
    """
    Computes CLIP embeddings for products using their TEXT descriptions (Name + Brand + Category).
    This is INSTANT (Seconds) compared to downloading images (Minutes/Hours).
    CLIP allows matching an Image Query against these Text Embeddings seamlessly.
    """
    model, processor = load_clip_model()
    
    # clear session state of any partial image loads if existing
    
    # 1. Prepare Text Data
    # We combine important textual features for a rich semantic representation
    # Fill NaNs to avoid errors
    data_filled = data.fillna('')
    
    # Construct descriptive text: "Brand Name Category Description"
    # Truncate descriptions to fit context window if needed, CLIP handles 77 tokens
    literals = (data_filled['Brand'] + " " + data_filled['Name'] + " " + data_filled['Category']).astype(str).tolist()
    
    # 2. Batch Process Text
    batch_features = []
    valid_indices = []
    batch_size = 64 # Text acts faster, can handle larger batches

    # Create a progress bar if running in Streamlit
    # st_progress = st.progress(0)
    
    total = len(literals)
    for i in range(0, total, batch_size):
        batch_text = literals[i:i+batch_size]
        batch_indices = data.index[i:i+batch_size].tolist()
        
        # Tokenize
        inputs = processor(text=batch_text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            raw_text_output = model.get_text_features(**inputs)
            text_features = extract_features_safe(raw_text_output)
            
            # Normalize for cosine similarity
            if isinstance(text_features, torch.Tensor):
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                batch_features.append(text_features.cpu().numpy())
                valid_indices.extend(batch_indices)
            
        # Optional: Update progress
        # if i + batch_size < total:
        #    st_progress.progress((i + batch_size) / total)
    
    # st_progress.empty()

    if batch_features:
        all_features = np.concatenate(batch_features, axis=0)
        return all_features, valid_indices
    return np.array([]), []


def recommend_by_image(uploaded_image, data=None, top_n=5):
    """
    Main function to be called from the Streamlit App.
    Returns a dataframe of recommended products.
    """
    # 1. Load Data if not provided
    if data is None:
        raw_data = get_data_from_firebase()
        if raw_data is None or raw_data.empty:
            st.error("No data available for recommendations.")
            return pd.DataFrame()
        data = process_data(raw_data)
    
    # 2. Get Features (Cached & Optimized to Text)
    # This will now accept the processed data and run in seconds
    with st.spinner("Indexing product catalog... (Fast Text-Mode)"):
        dataset_features, valid_indices = get_dataset_features(data)
    
    if len(dataset_features) == 0:
        st.error("Could not extract features from product catalog.")
        return pd.DataFrame()

    # 3. Process Query Image
    model, processor = load_clip_model()
    inputs = processor(images=uploaded_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get Image Features
        raw_output = model.get_image_features(**inputs)
        query_features = extract_features_safe(raw_output)
        
        # Ensure it's a tensor
        if not isinstance(query_features, torch.Tensor):
             st.error("Unexpected error in image feature extraction.")
             return pd.DataFrame()

        # Normalize
        query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
    
    query_features_np = query_features.cpu().numpy()

    # 4. Compute Similarity (Image Embedding vs Text Embeddings)
    # CLIP is trained for this Cross-Modal matching
    similarities = (dataset_features @ query_features_np.T).squeeze()
    
    # 5. Get Top N
    top_indices_local = similarities.argsort()[::-1][:top_n]
    
    # Map back to dataframe indices
    top_df_indices = [valid_indices[i] for i in top_indices_local]
    
    # Ensure indices are valid
    safe_indices = [i for i in top_df_indices if i in data.index]
    
    recommended_df = data.loc[safe_indices].copy()
    return recommended_df

# Placeholder function can be removed or kept as empty
def get_text_embeddings(text):
    return None
