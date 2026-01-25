import streamlit as st
import pandas as pd
import numpy as np
import random
import streamlit.components.v1 as components
from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations
from hybrid_approach import hybrid_recommendation_filtering
from item_based_collaborative_filtering import item_based_collaborative_filtering
st.set_page_config(page_title="AI based Ecommerce Recommendation system", layout="wide", page_icon="🛍️")
if "payment_done" not in st.session_state:
    st.session_state["payment_done"] = False

if "cart_items" not in st.session_state:
    st.session_state["cart_items"] = []

if st.query_params.get("payment") == "success":
    st.session_state["payment_done"] = True
    st.session_state["cart_items"] = []
    st.session_state["show_payment"] = False
    st.session_state["show_cart"] = False
    st.session_state["active_section"] = "Orders"
    st.query_params.clear()

url_category = st.query_params.get("category")
if url_category:
    st.session_state['search_input'] = url_category
    # Clear the query param so it doesn't stick
    # Using empty dictionary to clear all params is easiest or specific key
    # st.query_params.clear() # This might be too aggressive if we have user_id
    # Instead, we should pop it or set it to empty? 
    # Actually, let's just leave it or specific removal if supported. 
    # Recent streamlit supports typical dict operations on query_params.
    try:
        del st.query_params["category"]
    except:
        pass
st.markdown("""
<style>
    .block-container {
        padding-top: 15px;
        padding-bottom: 2rem;
    }
    div[data-testid="column"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    div[data-testid="column"]:hover {
        transform: translateY(-5px);
        box-shadow: 4px 4px 10px rgba(0,0,0,0.1);
    }
    /* Clickable Image Wrapper */
    .product-img-link {
        display: block;
        cursor: pointer;
    }
    .product-img {
        height: 200px;
        width: 100%;
        object-fit: contain;
        margin-bottom: 10px;
        border-radius: 4px;
        transition: transform 0.3s ease;
    }
    .product-img:hover {
        transform: scale(1.05);
    }
    /* Detail View specific styling */
    .detail-img {
        height: 400px;
        width: 100%;
        object-fit: contain;
        border-radius: 8px;
        border: 1px solid #eee;
    }
    .product-title {
        font-size: 14px;
        font-weight: 600;
        color: #333;
        margin-top: 10px;
        height: 40px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-brand {
        font-size: 12px;
        color: #777;
        margin-bottom: 5px;
    }
    .product-rating {
        color: #ffa41c;
        font-size: 14px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 5px;
        margin-bottom: 5px;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 5px;
    }
    /* Custom Header Styling */
    .title-text {
        font-size: 40px !important;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        padding-top: 20px;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    .horizontal-scroll-wrapper {
        position: relative;
        display: flex;
        align-items: center;
    }
    .horizontal-scroll-container {
        display: flex;
        overflow-x: auto;
        gap: 20px;
        padding: 10px 0;
        scroll-behavior: smooth;
        scrollbar-width: none; /* Firefox */
        -ms-overflow-style: none; /* IE 10+ */
    }
    .horizontal-scroll-container::-webkit-scrollbar {
        display: none; /* Chrome/Safari */
    }
    .scroll-btn {
        background-color: rgba(255, 255, 255, 0.8);
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        font-size: 20px;
        color: #333;
        cursor: pointer;
        z-index: 10;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        justify-content: center;
        align-items: center;
        transition: background 0.3s, transform 0.2s;
        position: absolute;
    }
    .scroll-btn:hover {
        background-color: #fff;
        transform: scale(1.1);
    }
    .scroll-btn.left {
        left: 10px;
    }
    .scroll-btn.right {
        right: 10px;
    }
    .cat-img {
        height: 180px;
        width: 100%;
        object-fit: cover;
        border-radius: 15px;
        transition: filter 0.3s ease, transform 0.3s ease;
        border: none;
        box-shadow: none;
        filter: blur(2px) brightness(0.7); /* Blur added as requested */
    }
    .cat-container:hover .cat-img {
        transform: scale(1.03);
        filter: blur(0px) brightness(0.6); /* Unblur on hover? Or keep blur? User said "background image should be slightly blurred". Let's keep it blurred but maybe less dark. */
    }
    .cat-label {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white; /* White text */
        font-weight: 800;
        font-size: 24px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.8);
        width: 100%;
        text-align: center;
        margin-top: 0;
        pointer-events: none;
    }
    .cat-container {
        position: relative;
        display: block;
        width: 100%;       /* Fluid width for grid columns */
        /* min-width: 300px; REMOVED to prevent overlap in grid */
        /* flex: 0 0 auto;   REMOVED as not needed for grid */
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-decoration: none;
        margin-bottom: 0; /* No bottom margin needed in scroll row */
    }
    /* Smart Badge Styling */
    .badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #ff4b2b;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: bold;
        z-index: 10;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .badge-value {
        background-color: #00b894;
    }
    .product-card-container {
        position: relative; /* For badge positioning */
    }
    /* Cart Table Styling */
    .cart-row {
        background-color: #fff;
        padding: 10px;
        border-bottom: 1px solid #eee;
        border-radius: 5px;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
    }
    .cart-img {
        height: 60px;
        width: 60px;
        object-fit: cover;
        border-radius: 5px;
        margin-right: 15px;
    }
    .checkout-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #eee;
    }
    .wishlist-icon {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 10;
        font-size: 20px;
        text-decoration: none;
        cursor: pointer;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        justify-content: center;
        align-items: center;
        transition: transform 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .wishlist-icon:hover {
        transform: scale(1.1);
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .wishlist-icon.active {
        color: #ff4b4b;
    }
    .wishlist-icon.inactive {
        color: #ccc;
    }
    .product-card-container {
        position: relative;
    }
</style>
""", unsafe_allow_html=True)
if 'selected_product' not in st.session_state:
    st.session_state['selected_product'] = None
def set_selected_product(product):
    """Callback to set the selected product in session state."""
    st.session_state['selected_product'] = product
def toggle_wishlist_func(prod_id):
    """Callback to toggle wishlist item without page reload."""
    target_uid = st.session_state.get('target_user_id', 0)
    if 'wishlists' not in st.session_state:
        st.session_state['wishlists'] = {}
    if target_uid not in st.session_state['wishlists']:
        st.session_state['wishlists'][target_uid] = []
    user_list = st.session_state['wishlists'][target_uid]
    if prod_id in user_list:
        user_list.remove(prod_id)
    else:
        user_list.append(prod_id)
def clear_query_params():
    """Clears query params to prevent sticking to the detail view on refresh."""
    current_uid = st.query_params.get("user_id")
    st.query_params.clear()
    if current_uid:
        st.query_params["user_id"] = current_uid
@st.cache_data
def load_and_process_data():
    """Loads and processes the dataset once."""
    try:
        raw_data = pd.read_csv("clean_data.csv")
        data = process_data(raw_data)
        
        # Ensure ProdID is consistent (int)
        if 'ProdID' in data.columns:
            data = data.dropna(subset=['ProdID'])
            data['ProdID'] = pd.to_numeric(data['ProdID'], errors='coerce')
            data = data.dropna(subset=['ProdID'])
            data['ProdID'] = data['ProdID'].astype(int)

        if 'ImageURL' in data.columns:
            data['ImageURL'] = data['ImageURL'].astype(str)
        return data
    except FileNotFoundError:
        st.error("Error: 'clean_data.csv' not found.")
        return None
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None
def get_smart_placeholder(name, prod_id):
    """Returns a high-quality placeholder image based on product keywords."""
    name_lower = str(name).lower()
    placeholders = {
        'nail': [
            "https://images.unsplash.com/photo-1632516643720-e7f5d7d6ecc9?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1604654894610-df63bc536371?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1519014816548-bf5fe059e98b?auto=format&fit=crop&w=400&q=80",
        ],
        'shampoo': [
            "https://images.unsplash.com/photo-1631729371254-42c2892f0e6e?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1556228720-19277026dfb6?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1585232351009-3135dfeb7e38?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1535585209827-a15fcdbc4c2d?auto=format&fit=crop&w=400&q=80",
        ],
        'conditioner': [
             "https://images.unsplash.com/photo-1576426863848-c218516d9b1a?auto=format&fit=crop&w=400&q=80",
             "https://images.unsplash.com/photo-1629198688000-71f23e745b6e?auto=format&fit=crop&w=400&q=80",
        ],
        'makeup': [
            "https://images.unsplash.com/photo-1596462502278-27bfdd403348?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1522335789203-abd652396e00?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1616683693504-3ea7e9ad6fec?auto=format&fit=crop&w=400&q=80",
        ],
        'generic': [
            "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1616940842431-c426588288d0?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1608248597279-f99d160bfbc8?auto=format&fit=crop&w=400&q=80",
            "https://images.unsplash.com/photo-1571781535009-ff1a3b538333?auto=format&fit=crop&w=400&q=80",
        ]
    }
    category = 'generic'
    if 'nail' in name_lower or 'lacquer' in name_lower or 'polish' in name_lower:
        category = 'nail'
    elif 'shampoo' in name_lower or 'wash' in name_lower:
        category = 'shampoo'
    elif 'conditioner' in name_lower or 'mask' in name_lower:
        category = 'conditioner'
    elif 'lip' in name_lower or 'eye' in name_lower or 'powder' in name_lower or 'up' in name_lower:
        category = 'makeup'
    try:
        seed_val = int(str(prod_id).replace("-", "").replace(" ", "")[:8], 16) if isinstance(prod_id, str) else int(prod_id)
    except:
        seed_val = hash(str(prod_id))
    random.seed(seed_val)
    images = placeholders.get(category, placeholders['generic'])
    selected_image = random.choice(images)
    random.seed(None)
    return selected_image
def get_product_image_url(product_row):
    """Refactored to be reusable."""
    image_url = product_row.get('ImageURL', '')
    if pd.isna(image_url) or image_url == '' or str(image_url).lower() == 'nan' or 'placehold.co' in str(image_url):
         prod_id = product_row.get('ProdID', 'default')
         prod_name = product_row.get('Name', '')
         image_url = get_smart_placeholder(prod_name, prod_id)
    return image_url
def sort_by_rating(df):
    """Helper to sort any dataframe by Rating (High -> Low)."""
    if df is not None and not df.empty and 'Rating' in df.columns:
        return df.sort_values(by='Rating', ascending=False)
    return df
def view_cart():
    """Renders the Cart Page."""
    st.markdown("<div class='section-header'>🛒 Your Shopping Cart</div>", unsafe_allow_html=True)
    # Back button removed as per request
    cart_items = st.session_state.get('cart_items', [])
    if not cart_items:
        st.info("Your cart is empty! Time to shop! 🛍️")
        return
    col_list, col_summary = st.columns([2, 1])
    total_price = 0
    with col_list:
        for i, item in enumerate(cart_items):
            price = 29.99
            total_price += price
            img_url = item.get('ImageURL', '')
            if pd.isna(img_url) or img_url == '':
                 img_url = "https://via.placeholder.com/60"
            st.markdown(
                f"""
                <div class="cart-row">
                    <img src="{img_url}" class="cart-img">
                    <div style="flex-grow: 1;">
                        <div style="font-weight: bold;">{item.get('Name')}</div>
                        <div style="color: #666; font-size: 12px;">{item.get('Brand')}</div>
                    </div>
                    <div style="font-weight: bold; margin-right: 15px;">${price}</div>
                </div>
                """, unsafe_allow_html=True
            )
            if st.button("Remove", key=f"rm_{i}"):
                st.session_state['cart_items'].pop(i)
                st.rerun()
    with col_summary:
        st.markdown(
            f"""
            <div class="checkout-box">
                <h3>Order Summary</h3>
                <div style="display: flex; justify-content: space-between;">
                    <span>Subtotal</span>
                    <span>${total_price:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Tax (5%)</span>
                    <span>${total_price*0.05:.2f}</span>
                </div>
                <hr>
                <div style="display: flex; justify-content: space-between; font-weight: bold; font-size: 18px;">
                    <span>Total</span>
                    <span>${total_price*1.05:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("")
        if st.button("Proceed to Checkout 💳", use_container_width=True):
            st.session_state["show_payment"] = True
def show_payment():
    st.markdown("<div class='section-header'>💳 Payment</div>", unsafe_allow_html=True)

    cart_items = st.session_state.get("cart_items", [])
    if not cart_items:
        st.warning("Cart is empty.")
        return

    # Demo total calculation
    total_amount = len(cart_items) * 499  # demo price
    st.subheader(f"Amount to Pay: ₹{total_amount}")

    components.html(f"""
    <script src="https://checkout.razorpay.com/v1/checkout.js"></script>

    <button id="rzp-button"
    style="
    padding:14px 28px;
    background:#0d6efd;
    color:white;
    border:none;
    border-radius:6px;
    font-size:16px;
    cursor:pointer;">
    Pay with Razorpay (Test)
    </button>

    <script>
    var options = {{
        "key": "rzp_test_S80byHh8aVKzjq",
        "amount": "{total_amount * 100}",
        "currency": "INR",
        "name": "AI E-Commerce Demo",
        "description": "Test Payment",
        "handler": function (response) {{
            alert("Payment Successful!\\nPayment ID: " + response.razorpay_payment_id);
            window.location.search = "?payment=success";
        }}
    }};
    var rzp = new Razorpay(options);
    document.getElementById("rzp-button").onclick = function(e) {{
        rzp.open();
        e.preventDefault();
    }};
    </script>
    """, height=700)


def view_product_details(product_row, data):
    """Renders the detailed view of a selected product."""
    components.html("""
        <script>
            function enforceScrollTop() {
                var targets = [
                    window.parent.document.querySelector('[data-testid="stAppViewContainer"]'),
                    window.parent.document.querySelector('section.main'),
                    window.parent.document.documentElement,
                    window.parent.document.body
                ];
                targets.forEach(function(target) {
                    if (target) {
                        target.scrollTop = 0;
                        target.scrollTo({top: 0, behavior: 'auto'});
                    }
                });
            }
            // 1. Immediate scroll
            enforceScrollTop();
            // 2. Continuous check for the first second (most aggressive)
            var checkCount = 0;
            var interval = setInterval(function() {
                enforceScrollTop();
                checkCount++;
                if (checkCount > 100) clearInterval(interval); // Stop after ~1s (10ms * 100)
            }, 10);
            // 3. MutationObserver to catch late DOM updates
            var observer = new MutationObserver(function(mutations) {
                enforceScrollTop();
            });
            var container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]') || window.parent.document.body;
            observer.observe(container, { childList: true, subtree: true, attributes: true });
            // Disconnect observer after 2 seconds to free resources
            setTimeout(function() {
                observer.disconnect();
                clearInterval(interval);
            }, 2000);
        </script>
    """, height=0)
    current_id = product_row['ProdID']
    def go_back():
        set_selected_product(None)
        clear_query_params()
    st.button("← Back to Shopping", on_click=go_back)
    st.markdown('<hr style="margin-top: 5px; margin-bottom: 5px; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        img_url = get_product_image_url(product_row)
        st.markdown(f'<img src="{img_url}" class="detail-img">', unsafe_allow_html=True)
    with col2:
        st.title(product_row.get('Name', 'Unknown Product'))
        st.subheader(product_row.get('Brand', 'Generic Brand'))
        rating = product_row.get('Rating', 0)
        stars = "⭐" * int(min(round(rating), 5))
        st.markdown(f"### {rating} {stars}")
        st.markdown("#### Product Description")
        st.write("Experience premium quality with this top-rated product. Perfect for your daily beauty routine.")
        st.markdown(f"#### Price: **${random.randint(15, 60)}.99**")
        if st.button("Add to Cart", key="btn_detail_add"):
            st.session_state['cart_items'].append(product_row.to_dict())
            st.toast(f"Added {product_row.get('Name')[:20]}... to cart! 🛒 ({len(st.session_state['cart_items'])})")
    st.markdown('<hr style="margin-top: 5px; margin-bottom: 5px; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)
    st.markdown("<div class='section-header'>✨ Similar Items</div>", unsafe_allow_html=True)
    try:
        similar_items = content_based_recommendation(data, item_name=product_row['Name'], top_n=4)
        similar_items = sort_by_rating(similar_items)
        display_product_grid(similar_items, section_key="detail_rec_content")
    except Exception as e:
        st.info("No similar products found.")
    st.markdown("<div class='section-header'>👥 Users Also Bought/Liked</div>", unsafe_allow_html=True)
    item_collab_recs = pd.DataFrame()
    try:
        prod_id = product_row.get('ProdID')
        if prod_id:
            item_collab_recs = item_based_collaborative_filtering(data, product_id=prod_id, top_n=4)
            item_collab_recs = sort_by_rating(item_collab_recs)
            if not item_collab_recs.empty:
                display_product_grid(item_collab_recs, section_key="detail_rec_item")
            else:
                st.info("Not enough data to see what others bought.")
        else:
             st.info("Product ID not available for recommendations.")
    except Exception as e:
         st.info("Not enough purchase history for this item.")
def display_product_card(product_row, key_suffix=""):
    """Displays a single product card."""
    image_url = get_product_image_url(product_row)
    
    # Safely extract ProdID
    prod_id = None
    if 'ProdID' in product_row and pd.notna(product_row['ProdID']):
        try:
            prod_id = int(product_row['ProdID'])
        except:
            prod_id = None

    badge_html = ""
    rating_val = product_row.get('Rating', 0)
    if rating_val >= 4.5:
         badge_html = "<div class='badge'>🏆 Top Rated</div>"
    elif rating_val >= 4.0:
         badge_html = "<div class='badge badge-value'>✨ Great Value</div>"
    
    target_uid = st.session_state.get('target_user_id', 0)
    user_wishlist = st.session_state.get('wishlists', {}).get(target_uid, [])
    
    # Heart Button Logic
    c_spacer, c_heart = st.columns([0.8, 0.2])
    with c_heart:
        if prod_id is not None:
            is_in_wishlist = prod_id in user_wishlist
            heart_symbol = "❤️" if is_in_wishlist else "🤍"
            st.button(heart_symbol, key=f"wish_btn_{prod_id}_{key_suffix}", on_click=toggle_wishlist_func, args=(prod_id,), help="Add to Wishlist")
        else:
            # Placeholder if no ID
            st.write("🤍")

    st.markdown(
        f'<div class="product-card-container">'
        f'{badge_html}'
        f'<img src="{image_url}" class="product-img">'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"<div class='product-title'>{product_row.get('Name', 'Unknown Product')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='product-brand'>{product_row.get('Brand', 'Generic')}</div>", unsafe_allow_html=True)
    rating = product_row.get('Rating', 0)
    stars = "⭐" * int(min(round(rating), 5))
    st.markdown(f"<div class='product-rating'>{rating} {stars}</div>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        # Only show details if we have an ID or valid row
        if prod_id is not None:
             st.button("Details",
                     key=f"btn_det_{prod_id}_{key_suffix}",
                     on_click=set_selected_product,
                     args=(product_row,))
        else:
             st.button("Details", key=f"btn_det_fail_{key_suffix}", disabled=True)

    with col_btn2:
        if st.button("Add", key=f"btn_add_{prod_id if prod_id else 'nan'}_{key_suffix}"):
            st.session_state['cart_items'].append(product_row.to_dict())
            st.toast(f"Added to Cart! 🛒 ({len(st.session_state['cart_items'])})")
            st.rerun()
def display_product_grid(products_df, section_key):
    """Renders a grid of product cards."""
    if products_df is None or products_df.empty:
        st.info("No products found.")
        return
    cols = st.columns(4)
    indices = products_df.index.tolist()
    for i, idx in enumerate(indices):
        col = cols[i % 4]
        with col:
            display_product_card(products_df.loc[idx], key_suffix=f"{section_key}_{i}")

def login_page(data):
    """Renders the Login and Signup page."""
    st.markdown("""
        <style>
            .login-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 30px;
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stButton>button {
                width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="title-text" style="text-align: center;">Welcome Back!</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666;">Please login to continue</p>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                user_id_input = st.text_input("User ID", placeholder="Enter your User ID")
                password_input = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    if not user_id_input:
                        st.error("Please enter a User ID.")
                    elif password_input != "infosys@123":
                        st.error("Incorrect password.")
                    else:
                        try:
                            uid = int(user_id_input)
                            if uid in data['ID'].values:
                                st.session_state['logged_in'] = True
                                st.session_state['target_user_id'] = uid
                                st.query_params['user_id'] = str(uid) # Persist to URL
                                st.success("Login Successful!")
                                st.rerun()
                            else:
                                st.error("User ID not found.")
                        except ValueError:
                            st.error("User ID must be a number.")
        
        with tab2:
            st.markdown("### New here?")
            st.write("Create a new account effortlessly.")
            if st.button("Create New Account"):
                try:
                    max_id = data['ID'].max()
                    if pd.isna(max_id):
                        new_id = 1
                    else:
                        new_id = int(max_id) + 1
                    
                    # In a real app, we would save this to the CSV/Database
                    # For now, we simulate it in session for this user
                    # Note: This checks 'clean_data.csv' on disk, so 'data' needs to be fresh
                    
                    st.session_state['logged_in'] = True
                    st.session_state['target_user_id'] = new_id
                    st.query_params['user_id'] = str(new_id) # Persist to URL
                    st.success(f"Account Created! Your User ID is **{new_id}**. Please remember it.")
                    st.info(f"Your default password is 'infosys@123'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating account: {e}")


def main():
    data = load_and_process_data()
    if data is None:
        return

    # Authentication Check & Session Restoration
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    # Try to restore session from URL if not logged in
    if not st.session_state['logged_in']:
        qp_uid = st.query_params.get("user_id")
        if qp_uid:
            try:
                restored_uid = int(qp_uid)
                if restored_uid in data['ID'].values:
                     st.session_state['logged_in'] = True
                     st.session_state['target_user_id'] = restored_uid
            except:
                pass

    if not st.session_state['logged_in']:
        login_page(data)
        return
    if st.session_state.get("payment_done"):
        st.success("🎉 Payment successful! Your order has been placed.")
        st.session_state["payment_done"] = False

    if 'Price' not in data.columns:
        np.random.seed(42)                                       
        data['Price'] = np.random.uniform(15.0, 100.0, size=len(data)).round(2)
    if 'cart_items' not in st.session_state:
        st.session_state['cart_items'] = []
    if 'wishlists' not in st.session_state:
        st.session_state['wishlists'] = {}                         
    try:
        query_params = st.query_params
        if 'prev_q_param' not in st.session_state:
             st.session_state['prev_q_param'] = ""

        # Sync URL -> Input (Only if URL changed)
        current_q_param = query_params.get("category") or query_params.get("q") or ""
        current_q_param = str(current_q_param).strip()
        
        if current_q_param != st.session_state['prev_q_param']:
             if current_q_param:
                 st.session_state['search_input'] = current_q_param
             st.session_state['prev_q_param'] = current_q_param
             # If URL changed (navigated), trust the page param
        
        q_prod_id = query_params.get("product_id")
        if q_prod_id:
             q_prod_id = str(q_prod_id).strip()
             found_product = data[data['ProdID'].astype(str) == q_prod_id]
             if not found_product.empty:
                 st.session_state['selected_product'] = found_product.iloc[0]
                 st.toast(f"Auto-loading Product: {q_prod_id}")
                 if hasattr(st, 'query_params'):
                     st.query_params.clear()
                 elif hasattr(st, 'experimental_set_query_params'):
                     st.experimental_set_query_params()
    except Exception as e:
        st.error(f"Error parsing query params: {e}")
        try:
             q_params = st.experimental_get_query_params()
             q_prod_id = q_params.get("product_id", [None])[0]
             if q_prod_id:
                  found_product = data[data['ProdID'].astype(str) == str(q_prod_id)]
                  if not found_product.empty:
                       st.session_state['selected_product'] = found_product.iloc[0]
        except:
             pass

    # --- Title & Search Header ---
    h_col1, h_col2 = st.columns([3, 1]) # Ratio to give Title more space, Search approx 300px logic
    with h_col1:
        st.markdown('<h1 class="title-text" style="text-align: left; margin-bottom: 5px;">AI-Based E-commerce Recommendation System</h1>', unsafe_allow_html=True)
    with h_col2:
        # Align search bar to match title visual baseline
        st.markdown('<style>div[data-testid="stTextInput"] { width: 300px; margin-left: auto; margin-top: 20px; }</style>', unsafe_allow_html=True)
        # Spacer removed for better alignment
        if 'search_input' not in st.session_state:
            st.session_state['search_input'] = ""
        search_query = st.text_input("Search", value=st.session_state['search_input'], placeholder="🔍 Search...", label_visibility="collapsed", key="search_widget_header")
        if search_query: st.session_state['search_input'] = search_query
    
    # --- Top Navigation Header ---
    target_user_id = st.session_state.get('target_user_id', 0)
    
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
    with nav_col1:
         if st.button("🏠 Home", use_container_width=True):
             set_selected_product(None)
             st.session_state['search_input'] = ""
             st.session_state['active_section'] = 'Home'
             st.session_state['show_cart'] = False # Ensure we exit cart view
             st.session_state['show_payment'] = False 
             clear_query_params()
             st.rerun()
    with nav_col2:
         if st.button("❤️ Wishlist", use_container_width=True):
             set_selected_product(None)
             st.session_state['active_section'] = 'Wishlist'
             st.session_state['show_cart'] = False 
             st.session_state['show_payment'] = False
             st.rerun()
    with nav_col3:
         if st.button("📦 Orders", use_container_width=True):
             set_selected_product(None)
             st.session_state['active_section'] = 'Orders'
             st.session_state['show_cart'] = False 
             st.rerun()
    with nav_col4:
        # We can also put Cart here or keep it in the search row. 
        # Including it here for consistent "Navigation" experience.
        c_count = len(st.session_state.get('cart_items', []))
        if st.button(f"🛒 Cart ({c_count})", key="nav_cart_header", use_container_width=True):
             set_selected_product(None)
             st.session_state['active_section'] = 'Cart' # Explicit state for clarity
             st.session_state['show_cart'] = True
             st.session_state['show_payment'] = False
             st.rerun()
    with nav_col5:
         if st.button("👤 Profile", key="profile_header", use_container_width=True):
             set_selected_product(None)
             st.session_state['active_section'] = 'Profile'
             st.session_state['show_cart'] = False
             st.session_state['show_payment'] = False
             st.rerun()

    st.markdown('<hr style="margin-top: 5px; margin-bottom: 5px; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)

    # --- Profile Section in Main Area (if active) ---
    if st.session_state.get('active_section') == 'Profile':
        st.markdown(f"### User Profile: #{target_user_id}")
        st.info(f"You are logged in as User #{target_user_id}")
        if st.button("Logout", key="logout_profile_btn"):
             st.session_state['logged_in'] = False
             st.session_state['target_user_id'] = 0
             st.session_state['active_section'] = 'Home'
             st.query_params.clear() # Clear URL persistence
             st.rerun()
        st.divider()

    # --- Filters (Main Area) ---
    # Moved from Sidebar to Expander
    show_filters = False
    search_active = st.session_state.get('search_input', '') != ''
    selected_brands = []
    min_rating = 0.0
    sort_option = "Relevance"
    
    if search_active:
         show_filters = True
    
    filtered_data = data.copy()
    
    if show_filters:
        # Pushing filters down a bit if needed or keeping compact
        with st.expander("🔍 Filter & Sort Options", expanded=False):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                sort_option = st.selectbox("Sort By", ["Relevance", "Price: Low to High", "Price: High to Low", "Rating: High to Low"])
            with f_col2:
                 all_brands = sorted(data['Brand'].dropna().unique().tolist())
                 selected_brands = st.multiselect("Brand", all_brands)
            with f_col3:
                 min_rating = st.slider("Min Rating", 0.0, 5.0, 3.0, 0.5)

    if selected_brands:
        filtered_data = filtered_data[filtered_data['Brand'].isin(selected_brands)]
    filtered_data = filtered_data[filtered_data['Rating'] >= min_rating]
    
    if sort_option == "Price: Low to High":
            filtered_data = filtered_data.sort_values(by='Price', ascending=True)
    elif sort_option == "Price: High to Low":
            filtered_data = filtered_data.sort_values(by='Price', ascending=False)
    elif sort_option == "Rating: High to Low":
            filtered_data = filtered_data.sort_values(by='Rating', ascending=False)
            
    is_filtering = (selected_brands or min_rating > 3.0) and not search_active
    if st.session_state['selected_product'] is not None:
        view_product_details(st.session_state['selected_product'], data)
    else:
        # Removed search bar from here as it is moved to header
        pass
        if st.session_state.get("show_payment", False):
             show_payment()
        elif st.session_state.get('show_cart', False):
             view_cart()
        elif st.session_state.get("active_section") == "Orders":
             st.markdown(f"<div class='section-header'>📦 Your Orders (User ID: {target_user_id})</div>", unsafe_allow_html=True)
             if 'ID' in data.columns:
                 user_orders = data[data['ID'] == target_user_id]
                 if user_orders.empty:
                     st.info(f"No previous orders found for User {target_user_id}.")
                 else:
                     st.success(f"Found {len(user_orders)} past orders.")
                     display_product_grid(user_orders, section_key="orders")
             else:
                 st.error("Order data not available (ID column missing).")
        elif st.session_state.get("active_section") == "Wishlist":
             st.markdown(f"<div class='section-header'>❤️ Your Wishlist (User ID: {target_user_id})</div>", unsafe_allow_html=True)
             w_list = st.session_state.get('wishlists', {}).get(target_user_id, [])
             if not w_list:
                 st.info("Your wishlist is empty. Click the ❤️ on products to add them!")
             else:
                 wishlist_products = data[data['ProdID'].isin(w_list)]
                 # Fix: Deduplicate products because data file has multiple entries per prod (ratings)
                 wishlist_products = wishlist_products.drop_duplicates(subset=['ProdID'])
                 
                 if wishlist_products.empty:
                      st.warning("Wishlist items found, but product details are missing from database.")
                 else:
                      st.success(f"Found {len(wishlist_products)} items in your wishlist.")
                      display_product_grid(wishlist_products, section_key="wishlist")
        elif is_filtering:
             st.markdown(f"<div class='section-header'>🔍 Filtered Results ({len(filtered_data)})</div>", unsafe_allow_html=True)
             display_product_grid(filtered_data, section_key="filtered")
        elif search_query:
            st.markdown(f"<div class='section-header'>Results for '{search_query}'</div>", unsafe_allow_html=True)
            try:
                search_results = data[data['Name'].astype(str).str.contains(search_query, case=False, na=False)]
                if selected_brands:
                    search_results = search_results[search_results['Brand'].isin(selected_brands)]
                search_results = search_results[search_results['Rating'] >= min_rating]
                if sort_option == "Price: Low to High":
                     search_results = search_results.sort_values(by='Price', ascending=True)
                elif sort_option == "Price: High to Low":
                     search_results = search_results.sort_values(by='Price', ascending=False)
                elif sort_option == "Rating: High to Low":
                     search_results = search_results.sort_values(by='Rating', ascending=False)
                else:
                     search_results = sort_by_rating(search_results)
                if search_results.empty:
                    st.warning(f"No products found matching '{search_query}'. Trying hybrid recommendation...")
                    search_results = hybrid_recommendation_filtering(data, item_name=search_query, target_user_id=target_user_id, top_n=10)
                    search_results = sort_by_rating(search_results)
                    if search_results.empty:
                        st.error("No results found.")
                else:
                    st.success(f"Found {len(search_results)} items.")
                
                # Pagination Logic
                
                # Get current page from URL or default to 1
                try:
                    query_params = st.query_params
                    current_page = int(query_params.get("page", 1))
                except:
                    current_page = 1
                
                # If we are searching, ensure URL reflects it (Bidirectional Sync)
                # But don't overwrite if it's just a page change
                current_url_q = query_params.get("q", "")
                current_url_cat = query_params.get("category", "")
                
                # If the active search query differs from what's in the URL, it means the User typed it.
                # We should update URL and reset page to 1.
                active_q_in_url = current_url_cat if current_url_cat else current_url_q
                
                if search_query and search_query != str(active_q_in_url).strip():
                     # User typed new search
                     st.query_params["q"] = search_query
                     if "category" in st.query_params: del st.query_params["category"]
                     st.query_params["page"] = 1
                     current_page = 1
                     # Update our prev tracker so we don't re-read it next run
                     st.session_state['prev_q_param'] = search_query
                
                items_per_page = 20
                total_items = len(search_results)
                total_pages = (total_items + items_per_page - 1) // items_per_page if total_items > 0 else 1
                
                # Ensure current_page is valid
                if current_page < 1: current_page = 1
                if current_page > total_pages: current_page = total_pages
                
                start_idx = (current_page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                
                paginated_results = search_results.iloc[start_idx:end_idx]
                
                display_product_grid(paginated_results, section_key="search")
                
                if total_pages > 1:
                    st.markdown('<hr style="margin-top: 5px; margin-bottom: 5px; border: 0; border-top: 1px solid #eee;">', unsafe_allow_html=True)
                    
                    # Link-Based Pagination for Scroll Reset
                    start_p = max(1, min(current_page - 4, total_pages - 9) if total_pages > 9 else 1)
                    end_p = min(start_p + 9, total_pages)
                    page_range = range(start_p, end_p + 1)
                    
                    # Helper to generate link
                    # We need to preserve current params: user_id, search_query (as q or category)
                    # We construct the base URL query string
                    
                    base_params = []
                    if target_user_id:
                        base_params.append(f"user_id={target_user_id}")
                    
                    # Determine if we are searching by category or generic query
                    q_cat = st.query_params.get("category")
                    if q_cat:
                         base_params.append(f"category={q_cat}")
                    elif search_query:
                         # If it's a generic search, put it in 'q'
                         # We encode it simply here (simulated)
                         base_params.append(f"q={search_query}")

                    base_qs = "&".join(base_params)
                    
                    def make_link(p, text, active=False):
                        if active:
                            return f'<span style="padding: 5px 10px; border: 1px solid #ccc; margin: 0 2px; border-radius: 5px; background-color: #f0f0f0; color: #333; cursor: default;">{text}</span>'
                        
                        url = f"./?{base_qs}&page={p}"
                        return f'<a href="{url}" target="_self" style="text-decoration: none; padding: 5px 10px; border: 1px solid #eee; margin: 0 2px; border-radius: 5px; color: #d63031; font-weight: bold;">{text}</a>'

                    # Build HTML for pagination
                    pagination_html = '<div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">'
                    
                    # Prev
                    if current_page > 1:
                        pagination_html += make_link(current_page - 1, "◀")
                    else:
                        pagination_html += '<span style="padding: 5px 10px; margin: 0 2px; color: #ccc;">◀</span>'
                    
                    # Pages
                    for p_num in page_range:
                        is_current = (p_num == current_page)
                        pagination_html += make_link(p_num, str(p_num), active=is_current)
                        
                    # Next
                    if current_page < total_pages:
                        pagination_html += make_link(current_page + 1, "▶")
                    else:
                        pagination_html += '<span style="padding: 5px 10px; margin: 0 2px; color: #ccc;">▶</span>'
                    
                    pagination_html += '</div>'
                    
                    st.markdown(pagination_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Search error: {e}")
        else:
            st.markdown("<div class='section-header'>📦 Shop by Category</div>", unsafe_allow_html=True)
            
            categories = [
                {"name": "Household", "img": "https://mastcert.com/wp-content/uploads/2023/05/bytovaya-himiya.jpg?q=80&w=600&auto=format&fit=crop"},
                {"name": "Grooming", "img": "https://blogscdn.thehut.net/wp-content/uploads/sites/32/2018/04/17133037/1200x672_217775857-MC-MK-April-photography-batching-Shot14_1200x672_acf_cropped_1200x672_acf_cropped.jpg?q=80&w=600&auto=format&fit=crop"},
                {"name": "Fragrance", "img": "https://images.unsplash.com/photo-1592945403244-b3fbafd7f539?q=80&w=600&auto=format&fit=crop"},
                {"name": "Hair Care", "img": "https://images.unsplash.com/photo-1560869713-7d0a29430803?q=80&w=600&auto=format&fit=crop"},
                {"name": "Nail Polish", "img": "https://www.makeup.com/-/media/project/loreal/brand-sites/mdc/americas/us/articles/2023/01-january/04-does-nail-polish-expire/does-nail-polish-expire-hero-mudc-122722.jpg?cx=0.5&cy=0.5&cw=705&ch=529&blr=False&hash=92FB2BCC56C0A381A75826D5939CFF96?q=80&w=600&auto=format&fit=crop"},
                {"name": "Makeup", "img": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?q=80&w=600&auto=format&fit=crop"}
            ]
            
            # Render as horizontal scrolling HTML container with buttons
            # We use a unique ID for the container to target it with JS
            # Replaced persistent iframe with native Streamlit Grid for reliable clicking
            cat_cols = st.columns(6)
            for i, cat in enumerate(categories):
                with cat_cols[i % 6]:
                     cat_name = cat['name']
                     img_url = cat['img']
                     st.markdown(
                        f'<a href="./?category={cat_name}&user_id={target_user_id}" target="_self" class="cat-container">'
                        f'<img src="{img_url}" class="cat-img" title="Shop {cat_name}">'
                        f'<div class="cat-label">{cat_name}</div>'
                        f'</a>',
                        unsafe_allow_html=True
                     )
            # Only show recommendations if NOT searching
            if not search_active:
                if target_user_id == 0:
                    st.markdown("<div class='section-header'>🌟 Top Rated Products for New Customers</div>", unsafe_allow_html=True)
                    try:
                        top_rated_new = get_top_rated_items(data, top_n=8)
                        top_rated_new = sort_by_rating(top_rated_new)
                        display_product_grid(top_rated_new, section_key="new_cust_top")
                    except Exception as e:
                        st.error(f"Error fetching top rated items: {e}")
                else:
                    try:
                        user_history = data[data['ID'] == target_user_id]
                        if not user_history.empty:
                             user_history = user_history.drop_duplicates(subset=['ProdID'])
                             # User requested "min of 4", increasing limit to show more history if available
                             latest_rated = user_history.tail(4) 
                             latest_rated = latest_rated.iloc[::-1] 
                             st.markdown(f"<div class='section-header'>⭐ Previously Rated by You (User {target_user_id})</div>", unsafe_allow_html=True)
                             display_product_grid(latest_rated, section_key="prev_rated")
                    except Exception as e:
                        st.error(f"Error loading user history: {e}")
                
                st.markdown("<div class='section-header'>🔥 Best Sellers</div>", unsafe_allow_html=True)
                try:
                    top_rated = get_top_rated_items(data, top_n=4)
                    top_rated = sort_by_rating(top_rated)
                    display_product_grid(top_rated, section_key="top_rated")
                except:
                    pass
                if target_user_id!=0:
                    st.markdown(f"<div class='section-header'>💙 Recommended for You (User {target_user_id})</div>", unsafe_allow_html=True)
                try:
                    collab_recs = collaborative_filtering_recommendations(data, target_user_id=target_user_id, top_n=12)
                    collab_recs = sort_by_rating(collab_recs)
                    collab_recs = collab_recs.iloc[:12]
                    if not collab_recs.empty:
                        display_product_grid(collab_recs, section_key="collab")
                except:
                    pass
    
            # Top Deals also only show if not searching
            if not search_active:
                 st.markdown("<div class='section-header'>🔥 Top Deals of the Week</div>", unsafe_allow_html=True)
                 try:
                     if 'Price' in data.columns:
                         deals = data[(data['Rating'] >= 4.0) & (data['Price'] < 40.0)]
                         if len(deals) < 4:
                             deals = data[data['Price'] < 50.0].sort_values(by='Rating', ascending=False)
                         deals = deals.sort_values(by='Price', ascending=True).head(4)
                         if not deals.empty:
                             display_product_grid(deals, section_key="top_deals")
                         else:
                              st.info("No top deals available this week.")
                     else:
                          st.info("Deals are updating...")
                 except Exception as e:
                     pass
    st.markdown("---")
    st.caption("© 2024 ShopEasy E-Commerce Demo | Powered by Streamlit & Hybrid Recommendation System")
if __name__ == "__main__":
    main()
