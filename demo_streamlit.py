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
st.markdown("""
<style>
    .block-container {
        padding-top: 5rem;
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
        margin-top: 40px;
        margin-bottom: 20px;
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
        padding-top: 0px;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    .cat-img {
        height: 150px;
        width: 100%;
        object-fit: cover;
        border-radius: 50%;
        margin-bottom: 10px;
        transition: transform 0.3s ease;
        border: 2px solid #fff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cat-img:hover {
        transform: scale(1.1);
        border-color: #ff4b2b;
    }
    .cat-label {
        font-weight: bold;
        color: #333;
        margin-top: 10px;
        text-align: center;
        width: 100%;
        display: block;
        font-size: 14px;
    }
    .cat-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-decoration: none;
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
</style>
""", unsafe_allow_html=True)
if 'selected_product' not in st.session_state:
    st.session_state['selected_product'] = None
def set_selected_product(product):
    """Callback to set the selected product in session state."""
    st.session_state['selected_product'] = product
def clear_query_params():
    """Clears query params to prevent sticking to the detail view on refresh."""
    st.query_params.clear()
@st.cache_data
def load_and_process_data():
    """Loads and processes the dataset once."""
    try:
        raw_data = pd.read_csv("clean_data.csv")
        data = process_data(raw_data)
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
    if st.button("← Back to Shopping", key="back_cart"):
        st.session_state['show_cart'] = False
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
            st.balloons()
            st.toast("Order Placed Successfully! 🎉")
            st.session_state['cart_items'] = []
            st.session_state['show_cart'] = False
            st.rerun()
def view_product_details(product_row, data):
    """Renders the detailed view of a selected product."""
    # Robust scroll-to-top using MutationObserver to fight Streamlit's scroll restoration
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
    st.markdown("---")
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
    st.markdown("---")
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
    if 'ProdID' in product_row and pd.notna(product_row['ProdID']):
        prod_id = int(product_row['ProdID'])
        badge_html = ""
        rating_val = product_row.get('Rating', 0)
        if rating_val >= 4.5:
             badge_html = "<div class='badge'>🏆 Top Rated</div>"
        elif rating_val >= 4.0:
             badge_html = "<div class='badge badge-value'>✨ Great Value</div>"
        st.markdown(
            f'<div class="product-card-container">'
            f'{badge_html}'
            f'<img src="{image_url}" class="product-img">'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        prod_id = random.randint(0, 100000)
        st.markdown(f'<img src="{image_url}" class="product-img">', unsafe_allow_html=True)
    st.markdown(f"<div class='product-title'>{product_row.get('Name', 'Unknown Product')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='product-brand'>{product_row.get('Brand', 'Generic')}</div>", unsafe_allow_html=True)
    rating = product_row.get('Rating', 0)
    stars = "⭐" * int(min(round(rating), 5))
    st.markdown(f"<div class='product-rating'>{rating} {stars}</div>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.button("Details",
                 key=f"btn_det_{prod_id}_{key_suffix}",
                 on_click=set_selected_product,
                 args=(product_row,))
    with col_btn2:
        if st.button("Add", key=f"btn_add_{prod_id}_{key_suffix}"):
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
def main():
    data = load_and_process_data()
    if data is None:
        return
    if 'cart_items' not in st.session_state:
        st.session_state['cart_items'] = []
    try:
        query_params = st.query_params
        q_category = query_params.get("category")
        if q_category:
             st.session_state['search_input'] = str(q_category).strip()
             if hasattr(st, 'query_params'):
                 st.query_params.clear()
             elif hasattr(st, 'experimental_set_query_params'):
                 st.experimental_set_query_params()
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
    with st.sidebar:
        st.title("👤 Account")
        if 'target_user_id' not in st.session_state:
            st.session_state['target_user_id'] = 0  # Default value

        def update_user_id():
            st.session_state['target_user_id'] = st.session_state['user_id_widget']

        st.number_input(
            "User ID (Simulation)", 
            min_value=0, 
            step=1,
            value=st.session_state['target_user_id'],
            key="user_id_widget",
            on_change=update_user_id
        )
        target_user_id = st.session_state['target_user_id']
        st.divider()
        st.subheader("Navigation")
        if st.button("🏠 Home"):
            set_selected_product(None)
            clear_query_params()
            st.rerun()
        st.radio("Go to:", ["Wishlist", "Orders"], label_visibility="collapsed")
        st.divider()
        st.subheader("Filters")
        all_brands = sorted(data['Brand'].dropna().unique().tolist())
        selected_brands = st.multiselect("Brand", all_brands)
        min_rating = st.slider("Min Rating", 0.0, 5.0, 3.0, 0.5)
        filtered_data = data.copy()
        if selected_brands:
            filtered_data = filtered_data[filtered_data['Brand'].isin(selected_brands)]
        filtered_data = filtered_data[filtered_data['Rating'] >= min_rating]
        is_filtering = (selected_brands or min_rating > 3.0)
    if st.session_state['selected_product'] is not None:
        view_product_details(st.session_state['selected_product'], data)
    else:
        col1, col2, col3 = st.columns([3, 2, 0.5])
        with col1:
             st.markdown('<h1 class="title-text">AI-Based E-commerce<br>Recommendation System</h1>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="margin-top: 25px;"></div>', unsafe_allow_html=True)
            if 'search_input' not in st.session_state:
                st.session_state['search_input'] = ""
            search_query = st.text_input("Search products...", value=st.session_state['search_input'], placeholder="🔍 Search for 'Nail Polish', 'Shampoo'...", label_visibility="collapsed", key="search_widget")
            if search_query: st.session_state['search_input'] = search_query
        with col3:
             st.markdown('<div style="margin-top: 25px;"></div>', unsafe_allow_html=True)
             cart_count = len(st.session_state.get('cart_items', []))
             if st.button(f"🛒 {cart_count}", key="nav_cart_btn"):
                 st.session_state['show_cart'] = True
                 st.rerun()
        if st.session_state.get('show_cart', False):
             view_cart()
        elif is_filtering:
             st.markdown(f"<div class='section-header'>🔍 Filtered Results ({len(filtered_data)})</div>", unsafe_allow_html=True)
             display_product_grid(filtered_data, section_key="filtered")
        elif search_query:
            st.markdown(f"<div class='section-header'>Results for '{search_query}'</div>", unsafe_allow_html=True)
            try:
                search_results = data[data['Name'].astype(str).str.contains(search_query, case=False, na=False)]
                search_results = sort_by_rating(search_results)
                if search_results.empty:
                    st.warning(f"No products found matching '{search_query}'. Trying hybrid recommendation...")
                    search_results = hybrid_recommendation_filtering(data, item_name=search_query, target_user_id=target_user_id, top_n=10)
                    search_results = sort_by_rating(search_results)
                    if search_results.empty:
                        st.error("No results found.")
                else:
                    st.success(f"Found {len(search_results)} items.")
                display_product_grid(search_results, section_key="search")
            except Exception as e:
                st.error(f"Search error: {e}")
        else:
            st.markdown("<div class='section-header'>📦 Shop by Category</div>", unsafe_allow_html=True)
            cat_cols = st.columns(6)
            categories = [
                {"name": "Nail Polish", "img": "https://images.unsplash.com/photo-1604654894610-df63bc536371?q=80&w=300&auto=format&fit=crop"},
                {"name": "Skin Care", "img": "https://images.unsplash.com/photo-1598440947619-2c35fc9aa908?q=80&w=300&auto=format&fit=crop"},
                {"name": "Hair Care", "img": "https://images.unsplash.com/photo-1562322140-8baeececf3df?q=80&w=300&auto=format&fit=crop"},
                {"name": "Makeup", "img": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?q=80&w=300&auto=format&fit=crop"},
                {"name": "Fragrance", "img": "https://images.unsplash.com/photo-1541643600914-78b084683601?q=80&w=300&auto=format&fit=crop"},
                {"name": "Lips", "img": "https://images.unsplash.com/photo-1586495777744-4413f21062fa?q=80&w=300&auto=format&fit=crop"}
            ]
            for i, cat in enumerate(categories):
                with cat_cols[i]:
                    cat_name = cat['name']
                    img_url = cat['img']
                    st.markdown(
                        f'<a href="./?category={cat_name}" target="_self" class="cat-container">'
                        f'<img src="{img_url}" class="cat-img" title="Shop {cat_name}">'
                        f'<div class="cat-label">{cat_name}</div>'
                        f'</a>',
                        unsafe_allow_html=True
                    )
            
            # Recently Viewed (Session History) - for ALL users


            if target_user_id == 0:
                st.markdown("<div class='section-header'>🌟 Top Rated Products for New Customers</div>", unsafe_allow_html=True)
                try:
                    top_rated_new = get_top_rated_items(data, top_n=8)
                    top_rated_new = sort_by_rating(top_rated_new)
                    display_product_grid(top_rated_new, section_key="new_cust_top")
                except Exception as e:
                    st.error(f"Error fetching top rated items: {e}")
            else:
                # Previously Rated (User History)
                st.markdown(f"<div class='section-header'>⭐ Previously Rated by You (User {target_user_id})</div>", unsafe_allow_html=True)
                try:
                    # Filter data for the current user and take the last 4 items (assuming order implies recency)
                    user_history = data[data['ID'] == target_user_id]
                    if not user_history.empty:
                         # Ensure we don't show duplicates if the user rated the same item multiple times (optional, but good practice)
                         user_history = user_history.drop_duplicates(subset=['ProdID'])
                         latest_rated = user_history.tail(4)
                         # Reverse to show most recent first if the csv is chronological
                         latest_rated = latest_rated.iloc[::-1] 
                         display_product_grid(latest_rated, section_key="prev_rated")
                    else:
                         st.info("You haven't rated any products yet.")
                except Exception as e:
                    st.error(f"Error loading user history: {e}")

                st.markdown("<div class='section-header'>🔥 Best Sellers</div>", unsafe_allow_html=True)
                try:
                    top_rated = get_top_rated_items(data, top_n=4)
                    top_rated = sort_by_rating(top_rated)
                    display_product_grid(top_rated, section_key="top_rated")
                except:
                    pass
                st.markdown(f"<div class='section-header'>💙 Recommended for You (User {target_user_id})</div>", unsafe_allow_html=True)
                try:
                    collab_recs = collaborative_filtering_recommendations(data, target_user_id=target_user_id, top_n=12)
                    collab_recs = sort_by_rating(collab_recs)
                    collab_recs = collab_recs.iloc[:12]
                    if not collab_recs.empty:
                        display_product_grid(collab_recs, section_key="collab")
                except:
                    pass
    st.markdown("---")
    st.caption("© 2024 ShopEasy E-Commerce Demo | Powered by Streamlit & Hybrid Recommendation System")
if __name__ == "__main__":
    main()