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
        
        self.model = genai.GenerativeModel(
            'models/gemini-flash-latest',  # ← UPDATED TO WORKING MODEL
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 150,
                "top_p": 0.9,
            }
        )
         
        self.data = data
        self.chat = None
        
    def get_system_prompt(self):
        """Create strict system prompt with topic restrictions."""
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
            if "SAFETY" in str(e).upper():
                return "I can only help with shopping questions. What product are you looking for? 🛍️"
            return "Sorry, I can only assist with product and shopping queries. How can I help you shop today?"

def render_chatbot_ui(data):
    """Render chatbot UI with floating button."""
    
    if 'chatbot_open' not in st.session_state:
        st.session_state.chatbot_open = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot_instance' not in st.session_state:
        st.session_state.chatbot_instance = None
    
    st.markdown("""
    <style>
    .stButton button[kind="primary"] {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        font-size: 24px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        z-index: 1000 !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .message {
        margin-bottom: 15px;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        align-items: flex-end;
    }
    
    .bot-message {
        align-items: flex-start;
    }
    
    .message-bubble {
        max-width: 80%;
        padding: 12px 16px;
        border-radius: 18px;
        word-wrap: break-word;
        font-size: 14px;
    }
    
    .user-message .message-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .bot-message .message-bubble {
        background: white;
        color: #333;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    if not st.session_state.chatbot_open:
        if st.button("💬", key="chat_toggle", help="Open Shopping Assistant", type="primary"):
            st.session_state.chatbot_open = True
            st.rerun()
    
    if st.session_state.chatbot_open:
        if st.session_state.chatbot_instance is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                st.error("⚠️ API key not found! Please set GOOGLE_API_KEY in .env file")
                if st.button("Close"):
                    st.session_state.chatbot_open = False
                    st.rerun()
                return
            
            try:
                st.session_state.chatbot_instance = EcommerceChatbot(api_key, data)
                st.session_state.chatbot_instance.start_chat()
                st.session_state.chat_history.append({
                    "role": "bot",
                    "message": "Hi! 👋 I help with product questions and shopping. What are you looking for?"
                })
            except Exception as e:
                st.error(f"Error initializing chatbot: {e}")
                if st.button("Close"):
                    st.session_state.chatbot_open = False
                    st.rerun()
                return
        
        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown("### 🛍️ Shopping Assistant")
                st.caption("Ask about products & shopping only")
            with col2:
                if st.button("✕", key="close_chat"):
                    st.session_state.chatbot_open = False
                    st.rerun()
            
            st.markdown("---")
            
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div class="message user-message">'
                            f'<div class="message-bubble">{msg["message"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="message bot-message">'
                            f'<div class="message-bubble">{msg["message"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            
            st.markdown("---")
            user_input = st.chat_input("Ask about products...")
            
            if user_input:
                st.session_state.chat_history.append({
                    "role": "user",
                    "message": user_input
                })
                
                try:
                    with st.spinner("💭"):
                        response = st.session_state.chatbot_instance.send_message(user_input)
                        st.session_state.chat_history.append({
                            "role": "bot",
                            "message": response
                        })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "message": "I can only help with shopping questions. What product can I help you find?"
                    })
                
                st.rerun()
            
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = [{
                    "role": "bot",
                    "message": "Hi! 👋 I help with product questions and shopping. What are you looking for?"
                }]
                st.rerun()
