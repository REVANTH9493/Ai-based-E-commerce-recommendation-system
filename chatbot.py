import streamlit as st
from huggingface_hub import InferenceClient
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EcommerceChatbot:
    def __init__(self, api_key, data):
        """Initialize chatbot with Hugging Face Inference API and product data."""
        # Using a reliable, fast model for e-commerce assistance
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
        self.client = InferenceClient(model=self.model_id, token=api_key)
        self.data = data
        self.history = []

    def get_system_prompt(self):
        """Create strict system prompt with topic restrictions."""
        if self.data.empty:
             return "You are a shopping assistant. No products are currently loaded."

        categories = self.data['Category'].value_counts().head(10).index.tolist()
        brands = self.data['Brand'].value_counts().head(10).index.tolist()
        
        system_prompt = f"""You are a helpful e-commerce shopping assistant ONLY for beauty and personal care products.

STRICT RULES:
1. Keep ALL responses under 50 words
2. Primarily focus on: products, shopping, beauty, personal care, orders, recommendations.
3. Be polite and professional. You ARE allowed to answer basic greetings (Hi, Hello) and polite small talk (How are you?).
4. If asked about entirely unrelated sensitive topics (politics, news, sports, coding), gracefully redirect: "I can only help with shopping and product questions. How can I assist with your purchase today?"
5. Be friendly but concise.

Available Categories: {', '.join(categories[:5])}
Top Brands: {', '.join(brands[:5])}
Total Products: {len(self.data)}"""
        return system_prompt
    
    def search_products(self, query):
        """Search products based on user query."""
        if self.data.empty:
            return pd.DataFrame()
            
        query_lower = query.lower()
        
        # Handle "Best Selling" / "Popular" queries explicitly
        popular_keywords = ['best', 'popular', 'top', 'trending', 'hot']
        if any(w in query_lower for w in popular_keywords):
            if 'Rating' in self.data.columns:
                 return self.data.sort_values(by='Rating', ascending=False).head(5)
            return self.data.head(5)

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
        """Initialize chat session history with system prompt."""
        system_prompt = self.get_system_prompt()
        self.history = [
            {"role": "system", "content": system_prompt}
        ]
        return system_prompt
    
    def send_message(self, user_message):
        """Send message with topic validation and get response via Hugging Face.
        Returns: (response_text, found_products_df)
        """
        
        # 4️⃣ Search Your Product Data FIRST
        products = self.search_products(user_message)
        
        # 6️⃣ Construct a Controlled Prompt (Relaxed for casual chat)
        # We only hard-block truly problematic topics
        off_topic_hard_block = [
            'news', 'politics', 'sports', 'code', 'program',
            'math', 'calculate', 'translate', 'history', 'science'
        ]
        
        if any(pattern in user_message.lower() for pattern in off_topic_hard_block):
            return "I can only help with shopping and product questions. How can I assist with your purchase today? 🛍️", pd.DataFrame()

        # 5️⃣ Convert Products into Text (Prompt Injection)
        if not products.empty:
            product_list = "\n".join([
                f"- {row['Name']} – ₹{row['Price']}" 
                for _, row in products.iterrows()
            ])
            prompt_context = f"Available products:\n{product_list}"
        else:
            prompt_context = "No products found in the catalog for this specific query."
        
        # Build prompt
        full_prompt = f"""[SYSTEM] You are a helpful e-commerce shopping assistant. 
Rules:
- Be polite and professional.
- Only answer small talk if the user explicitly asks "How are you?" or greets you first. Do NOT start every message with a greeting.
- ONLY recommend products from the provided list if shopping-related.
- If no products match a shopping query, say you can't find specific matches.
- Response must be under 50 words.

[CONTEXT]
{prompt_context}

[USER]
{user_message}

[ASSISTANT]
"""
        
        # Use local history for context but primarily rely on injected products
        self.history.append({"role": "user", "content": full_prompt})
        
        try:
            # 7️⃣ Call Hugging Face Mistral API
            completion = self.client.chat_completion(
                messages=[{"role": "user", "content": full_prompt}], # Fresh prompt for strict adherence
                max_tokens=150,
                temperature=0.1 # 7️⃣ low temperature
            )
            
            # 8️⃣/9️⃣ Extract Response
            response_text = completion.choices[0].message.content
            
            # Add to history
            self.history.append({"role": "assistant", "content": response_text})
            
            return response_text, products
            
        except Exception as e:
            if "429" in str(e):
                 return "Hugging Face is currently busy. Please try again in a moment.", pd.DataFrame()
            return "I'm having trouble connecting to my AI core right now. How can I help you shop manually?", pd.DataFrame()


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
            if "HF_TOKEN" in st.secrets:
                api_key = st.secrets["HF_TOKEN"]
            else:
                api_key = os.getenv("HF_TOKEN")
                
            if not api_key:
                st.error("Missing Hugging Face API Token (HF_TOKEN)")
            else:
                st.session_state.chatbot_instance = EcommerceChatbot(api_key, data)
                st.session_state.chatbot_instance.start_chat()
                if not st.session_state.chat_history:
                    st.session_state.chat_history.append({
                        "role": "bot", 
                        "message": "Hi! 👋 how can I help you shop!"
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
            .product-mini-card {
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 8px;
                margin-top: 5px;
                background: white;
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            .product-mini-card img {
                width: 100%;
                border-radius: 4px;
            }
            .product-mini-card-name {
                font-weight: bold;
                font-size: 11px;
                color: #333;
            }
            .product-mini-card-price {
                color: #ff4b4b;
                font-weight: bold;
                font-size: 11px;
            }
            section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-testid="input_container"] {
                min-height: 2.5rem !important;
                border: none !important;
                background: transparent !important;
            }
            .rec-img-fixed {
                width: 100%;
                height: 150px !important;
                object-fit: contain !important;
                border-radius: 8px;
                background-color: #fff;
                margin-bottom: 5px;
            }
            div[data-testid="InputInstructions"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Shopping Assistant")
    
    # Simple container for messages
    # Custom Scrollable Message Area (Standardized Height)
    all_msgs_html = ""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            all_msgs_html += f'<div style="background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin-bottom: 5px; text-align: right; color: #333; font-size: 13px;">{msg["message"]}</div>'
        else:
            # Bot Message
            bot_html = f'<div style="background-color: #f1f3f4; padding: 10px; border-radius: 10px; margin-bottom: 5px; text-align: left; color: #333; font-size: 13px;">'
            bot_html += f'<div>{msg["message"]}</div>'
            
            # 1️⃣1️⃣ Display Product Cards (Separately) - REMOVED per user request
            # if "products" in msg and not msg["products"].empty:
            #    ... (removed)
            
            bot_html += '</div>'
            all_msgs_html += bot_html
    
    st.markdown(
        f'<div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 10px; padding: 10px; background: #ffffff; margin-bottom: 15px;">'
        f'{all_msgs_html}'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    # Input
    with st.form(key="sidebar_chat_form", clear_on_submit=True, border=False):
        user_input = st.text_input("Ask about products...", placeholder="Shampoo for dry hair...", label_visibility="collapsed")
        submitted = st.form_submit_button("Send", use_container_width=True)

    # 1️⃣1️⃣/1️⃣2️⃣ Interactive Product Cards (Separately)
    # Render interactive cards for the last bot message if it has products
    if st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]
        if last_msg["role"] == "bot" and "products" in last_msg and not last_msg["products"].empty:
            st.markdown("---")
            st.markdown("##### 🛒 Recommended for you:")
            for _, row in last_msg["products"].head(2).iterrows():
                with st.expander(f"{row['Name'][:40]}...", expanded=True):
                    cols = st.columns([1, 2])
                    with cols[0]:
                        img_url = row.get('ImageURL', 'https://via.placeholder.com/150')
                        st.markdown(f'<img src="{img_url}" class="rec-img-fixed">', unsafe_allow_html=True)
                    with cols[1]:
                        st.write(f"**Price:** ₹{row['Price']}")
                        if st.button(f"Add to Cart", key=f"chat_add_{row['Name'][:10]}_{_}"):
                            if 'cart_items' not in st.session_state:
                                st.session_state['cart_items'] = []
                            st.session_state['cart_items'].append(row.to_dict())
                            st.success("Added!")
                            st.rerun()

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        
        if st.session_state.chatbot_instance:
            try:
                resp, found_prods = st.session_state.chatbot_instance.send_message(user_input)
                st.session_state.chat_history.append({
                    "role": "bot", 
                    "message": resp,
                    "products": found_prods
                })
            except Exception as e:
                st.session_state.chat_history.append({"role": "bot", "message": "Error connecting to AI."})
        st.rerun()