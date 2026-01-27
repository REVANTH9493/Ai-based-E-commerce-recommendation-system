
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pandas as pd
import streamlit as st
import os

# Singleton pattern for Firebase app to avoid "App already exists" errors in Streamlit
if not firebase_admin._apps:
    try:
        # Check for secrets first (Cloud/secure deployment)
        if "firebase" in st.secrets:
            # Streamlit secrets usually behave like a dict or can be converted to one
            service_account_info = dict(st.secrets["firebase"])
            
            # Initialize with secrets
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://ai-based-recommendation-55bcb-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
        
        # Fallback to local file
        elif os.path.exists('serviceAccountKey.json'):
            cred = credentials.Certificate('serviceAccountKey.json')
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://ai-based-recommendation-55bcb-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
        else:
            st.error("Firebase credentials not found (checked secrets and serviceAccountKey.json).")
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")

@st.cache_data(ttl=600)  # Cache for 10 minutes to avoid hitting DB on every interaction
def get_data_from_firebase():
    """
    Fetches product data from Firebase Realtime Database and returns a Pandas DataFrame.
    """
    try:
        ref = db.reference('/products')
        data = ref.get()
        
        if data:
            # Data comes back as a list of dicts or dict of dicts depends on keys
            # Since we uploaded a list of records (orient='records'), it should be a list
            if isinstance(data, list):
                # Filter out any None values (in case of sparse arrays)
                data = [item for item in data if item is not None]
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame.from_dict(data, orient='index')
            else:
                return None
            
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching data from Firebase: {e}")
        return None

def get_users_from_firebase():
    """
    Fetches all user accounts from Firebase /users node.
    Returns a dictionary of user_id -> user_data.
    """
    try:
        ref = db.reference('/users')
        users_data = ref.get()
        if users_data:
            return users_data
        return {}
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return {}

def save_user_to_firebase(user_data):
    """
    Saves or updates a user record in Firebase /users node.
    user_data must contain 'user_id'.
    """
    try:
        if 'user_id' not in user_data:
            return False
            
        user_id = str(user_data['user_id'])
        ref = db.reference(f'/users/{user_id}')
        ref.set(user_data)
        return True
    except Exception as e:
        st.error(f"Error saving user: {e}")
        return False
