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
    """Render chatbot UI with floating button."""
    if not visible:
        return

    if 'chatbot_open' not in st.session_state:
        st.session_state.chatbot_open = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot_instance' not in st.session_state:
        st.session_state.chatbot_instance = None
    if 'chat_input_val' not in st.session_state:
        st.session_state.chat_input_val = ""
    
    # CSS for fixed popup
    st.markdown("""
    <style>
    /* Trigger Button Styling */
    .chat-trigger-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1000;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Main Chat Container Styling */
    div[data-testid="stVerticalBlock"]:has(div.chatbot-marker) {
        position: fixed;
        bottom: 80px;
        right: 30px;
        width: 380px;
        height: 500px; /* Increased slightly for better proportions */
        max-height: 80vh;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        z-index: 1001;
        overflow: hidden;
        border: 1px solid #eee;
        padding: 0 !important;
        display: flex;
        flex-direction: column;
        gap: 0 !important; /* Force remove gaps */
    }
    
    /* Remove default Streamlit padding inside the popup container */
    div[data-testid="stVerticalBlock"]:has(div.chatbot-marker) > div {
        width: 100%;
    }
    div[data-testid="stVerticalBlock"]:has(div.chatbot-marker) > div[data-testid="element-container"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Chat Header */
    .chat-header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        font-weight: 600;
        font-size: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #eee;
    }
    
    /* Close Button in Header */
    .close-btn {
        cursor: pointer;
        color: white; 
        font-weight: bold;
        opacity: 0.8;
    }
    .close-btn:hover { opacity: 1; }

    /* Messages Area */
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        padding: 15px;
        background-color: #f8f9fa;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .message {
        display: flex;
        flex-direction: column;
        max-width: 85%;
    }
    
    .user-message {
        align-self: flex-end;
        align-items: flex-end;
    }
    
    .bot-message {
        align-self: flex-start;
        align-items: flex-start;
    }
    
    .message-bubble {
        padding: 10px 14px;
        border-radius: 16px;
        font-size: 14px;
        line-height: 1.4;
        box-shadow: none !important;
    }
    
    .user-message .message-bubble {
        background: #667eea;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .bot-message .message-bubble {
        background: white;
        color: #333;
        border-bottom-left-radius: 4px;
        border: 1px solid #f0f0f0;
    }
    
    /* Hide default Streamlit elements that add spacing */
    div[data-testid="stVerticalBlock"]:has(div.chatbot-marker) iframe,
    div[data-testid="stVerticalBlock"]:has(div.chatbot-marker) .stMarkdown {
       margin-bottom: 0 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    # Render Trigger Button
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
    </style>
    """, unsafe_allow_html=True)
    
    if not st.session_state.chatbot_open:
        if st.button("💬", key="chat_toggle", help="Open Shopping Assistant", type="primary"):
            st.session_state.chatbot_open = True
            st.rerun()
    
    else: 
        # Chat Window
        with st.container():
            # Marker for CSS targeting - NOW INSIDE THE CONTAINER
            st.markdown('<div class="chatbot-marker"></div>', unsafe_allow_html=True)
            
            # Initialize Bot logic if needed
            if st.session_state.chatbot_instance is None:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    st.error("Missing API Key")
                else:
                    try:
                        st.session_state.chatbot_instance = EcommerceChatbot(api_key, data)
                        st.session_state.chatbot_instance.start_chat()
                        if not st.session_state.chat_history:
                            st.session_state.chat_history.append({
                                "role": "bot", 
                                "message": "Hi! 👋 available to help you shop!"
                            })
                    except Exception as e:
                        st.error(f"Error: {e}")

            # --- Custom HTML Header to avoid Streamlit Column Spacing ---
            # using st.columns usually adds gaps. We can try a pure HTML header row 
            # OR just styling the columns very tight. Let's stick to columns but remove their gaps via CSS above.
            
            # Header Row
            h1, h2 = st.columns([0.85, 0.15])
            with h1:
                st.markdown('<div style="font-weight: 600; color: #667eea; font-size: 25px; padding: 5px 0 0 15px;">🛍️ Assistant</div>', unsafe_allow_html=True)
            with h2:
                if st.button("✕", key="close_chat_btn"):
                    st.session_state.chatbot_open = False
                    st.rerun()
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # Spacer instead of HR

            # --- Messages ---
            # Using a fixed height container for scroll
            messages_container = st.container(height=350) 
            with messages_container:
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
            
            # --- Input Area ---
            # Using a container with background color to anchor it at bottom visually
            with st.container():
                st.markdown('<div class="input-container" style="padding: 10px; background-color: white; border-top: 1px solid #eee;">', unsafe_allow_html=True)
                
                with st.form(key="chat_input_form", clear_on_submit=True, border=False):
                    ic1, ic2 = st.columns([0.85, 0.15])
                    
                    with ic1:
                        st.text_input("Ask...", key="chat_form_input", placeholder="Type message...", label_visibility="collapsed")
                    
                    with ic2:
                        st.markdown("""
                        <style>
                        /* Style for the form submit button */
                        div[data-testid="stForm"] button {
                            padding: 0rem !important;
                            width: 38px !important;
                            height: 38px !important;
                            border-radius: 50% !important;
                            margin-top: 0px !important;
                            background-color: #667eea;
                            color: white;
                            border: none;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        div[data-testid="stForm"] button:hover {
                            background-color: #764ba2;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                        }
                        /* Remove default input styling to make it flush */
                        div[data-testid="stTextInput"] input {
                            padding: 0.5rem 1rem !important;
                            min-height: 2.5rem !important;
                            height: 2.5rem !important;
                            font-size: 14px !important;
                            border-radius: 20px !important;
                            border: 1px solid #eee !important;
                            background: #f9f9f9 !important;
                        }
                        div[data-testid="stTextInput"] div[data-testid="input_container"] {
                            min-height: 2.5rem !important;
                            border: none !important;
                            background: transparent !important;
                        }
                        
                        /* Remove form gap */
                        div[data-testid="stForm"] {
                            gap: 0px !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        submitted = st.form_submit_button("➤", help="Send")
                
                if submitted:
                    val = st.session_state.get("chat_form_input")
                    if val:
                        st.session_state.chat_history.append({"role": "user", "message": val})
                        # Response logic
                        if st.session_state.chatbot_instance:
                            try:
                                resp = st.session_state.chatbot_instance.send_message(val)
                                st.session_state.chat_history.append({"role": "bot", "message": resp})
                            except Exception as e:
                                st.session_state.chat_history.append({"role": "bot", "message": "Verify API Key."})
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)