import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EcommerceChatbot:
    def __init__(self, api_key, data):
        """Initialize chatbot with Gemini API and product data."""
        genai.configure(api_key=api_key)
        
        # --- FIXED MODEL NAME ---
        # gemini-1.5-flash is retired. 
        # gemini-2.5-flash is the current stable version with high rate limits.
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash', 
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2048,
                "top_p": 0.9,
            }
        )
         
        self.data = data
        self.chat = None

    def get_system_prompt(self):
        """Create strict system prompt with topic restrictions."""
        # Safety check for empty data
        if self.data.empty:
             return "You are a shopping assistant. No products are currently loaded."

        categories = self.data['Category'].value_counts().head(10).index.tolist()
        brands = self.data['Brand'].value_counts().head(10).index.tolist()
        
        system_prompt = f"""You are a helpful e-commerce shopping assistant ONLY for beauty and personal care products.

STRICT RULES:
1. Keep ALL responses under 50 words
2. ONLY discuss: products, shopping, beauty, personal care, orders, recommendations
3. If asked about unrelated topics (politics, news, coding, general knowledge, etc.), respond: "I can only help with shopping and product questions. How can I assist with your purchase today?"
4. Be friendly but concise
5. Never discuss topics outside e-commerce/shopping

Available Categories: {', '.join(categories[:5])}
Top Brands: {', '.join(brands[:5])}
Total Products: {len(self.data)}

Stay focused on helping customers shop!"""
        return system_prompt
    
    def search_products(self, query):
        """Search products based on user query."""
        if self.data.empty:
            return pd.DataFrame()
            
        query_lower = query.lower()
        mask = (
            self.data['Name'].str.lower().str.contains(query_lower, na=False) |
            self.data['Brand'].str.lower().str.contains(query_lower, na=False) |
            self.data['Category'].str.lower().str.contains(query_lower, na=False)
        )
        results = self.data[mask].head(3)
        return results
    
    def is_shopping_related(self, message):
        """Check if query is shopping/e-commerce related."""
        shopping_keywords = [
            'product', 'buy', 'purchase', 'price', 'recommend', 'shop', 'order',
            'brand', 'category', 'beauty', 'care', 'makeup', 'hair', 'skin',
            'nail', 'shampoo', 'lipstick', 'cream', 'oil', 'perfume', 'rating',
            'review', 'stock', 'available', 'compare', 'best', 'cheap', 'deal'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in shopping_keywords)
    
    def start_chat(self):
        """Initialize chat session with system prompt."""
        system_prompt = self.get_system_prompt()
        self.chat = self.model.start_chat(history=[])
        self.chat.send_message(system_prompt)
        return system_prompt
    
    def send_message(self, user_message):
        """Send message with topic validation and get response."""
        
        if not self.is_shopping_related(user_message):
            off_topic_patterns = [
                'news', 'weather', 'politics', 'sports', 'code', 'program',
                'math', 'calculate', 'translate', 'history', 'science', 'joke',
                'story', 'poem', 'song', 'movie', 'game', 'recipe'
            ]
            
            if any(pattern in user_message.lower() for pattern in off_topic_patterns):
                return "I can only help with shopping and product questions. How can I assist with your purchase today? 🛍️"
        
        products = self.search_products(user_message)
        
        if not products.empty:
            product_list = "\n".join([
                f"• {row['Name'][:50]} - {row['Brand']} (⭐{row['Rating']}/5)" 
                for _, row in products.head(3).iterrows()
            ])
            enhanced_message = f"""User: {user_message}

Found products:
{product_list}

Respond in MAX 50 words. Stay on topic (shopping/products only). Be helpful and concise."""
        else:
            enhanced_message = f"""User: {user_message}

Respond in MAX 50 words. ONLY discuss shopping/products. If off-topic, redirect to shopping. Be concise."""
        
        try:
            response = self.chat.send_message(enhanced_message)
            response_text = response.text
            
            words = response_text.split()
            if len(words) > 60:
                response_text = ' '.join(words[:60]) + "..."
            
            return response_text
            
        except Exception as e:
            if "429" in str(e):
                 return "I'm receiving too many requests right now. Please try again in a few seconds."
            if "SAFETY" in str(e).upper():
                return "I can only help with shopping questions. What product are you looking for? 🛍️"
            return "Sorry, I can only assist with product and shopping queries. How can I help you shop today?"


def render_chatbot_ui(data, visible=True):
    """Render chatbot UI embedded in sidebar."""
    if not visible:
        return

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot_instance' not in st.session_state:
        st.session_state.chatbot_instance = None
    
    # Initialize Bot logic if needed
    if st.session_state.chatbot_instance is None:
        try:
            # Try getting key from secrets first
            api_key = None
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
            else:
                api_key = os.getenv("GOOGLE_API_KEY")
                
            if not api_key:
                st.error("Missing Google API Key")
            else:
                st.session_state.chatbot_instance = EcommerceChatbot(api_key, data)
                st.session_state.chatbot_instance.start_chat()
                if not st.session_state.chat_history:
                    st.session_state.chat_history.append({
                        "role": "bot", 
                        "message": "Hi! 👋 available to help you shop!"
                    })
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Chat Interface (No Floating CSS) ---
    
    # Scoped CSS for Sidebar Inputs
    st.markdown("""
        <style>
            /* Remove default input styling to make it flush */
            section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
                padding: 0.5rem 1rem !important;
                min-height: 2.5rem !important;
                height: 2.5rem !important;
                font-size: 14px !important;
                border-radius: 20px !important;
                border: 1px solid #eee !important;
                background: #f9f9f9 !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-testid="input_container"] {
                min-height: 2.5rem !important;
                border: none !important;
                background: transparent !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Shopping Assistant")
    
    # Simple container for messages
    messages_container = st.container(height=500) 
    with messages_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin-bottom: 5px; text-align: right; color: #333;">{msg["message"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background-color: #f1f3f4; padding: 10px; border-radius: 10px; margin-bottom: 5px; text-align: left; color: #333;">{msg["message"]}</div>',
                    unsafe_allow_html=True
                )
    
    # Input
    with st.form(key="sidebar_chat_form", clear_on_submit=True, border=False):
        user_input = st.text_input("Ask about products...", placeholder="Shampoo for dry hair...", label_visibility="collapsed")
        submitted = st.form_submit_button("Send", use_container_width=True)
    
    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        
        if st.session_state.chatbot_instance:
            try:
                resp = st.session_state.chatbot_instance.send_message(user_input)
                st.session_state.chat_history.append({"role": "bot", "message": resp})
            except Exception as e:
                st.session_state.chat_history.append({"role": "bot", "message": "Error connecting to AI."})
        st.rerun()