import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Try to import surprise, handle failure gracefully
try:
    import surprise
except ImportError:
    surprise = None

# Set page configuration
st.set_page_config(page_title="Amazon-like Store", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Amazon-like styling
st.markdown("""
<style>
    .product-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
        height: 380px; /* Fixed height for alignment */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .product-image-placeholder {
        height: 120px;
        background-color: #f8f9fa;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .product-title {
        font-size: 14px;
        font-weight: bold;
        color: #0F1111;
        margin-bottom: 5px;
        height: 40px; /* Fixed height for 2 lines */
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-price {
        font-size: 18px;
        color: #B12704;
        font-weight: bold;
    }
    .prime-badge {
        color: #007600;
        font-weight: bold;
        font-size: 12px;
        margin-bottom: 5px;
    }
    .stButton>button {
        background-color: #ffd814;
        color: black;
        border: none;
        border-radius: 20px;
        padding: 5px 15px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #f7ca00;
    }
    .rec-container {
        background-color: #f3f3f3;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Helper for safe ID conversion
def safe_int(x):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return None

# Data Loading function
@st.cache_data
def load_data():
    error_log = []
    # Load Products
    try:
        products = pd.read_csv("products.csv")
        # Ensure product_id is integer
        products['product_id'] = products['product_id'].apply(safe_int)
        products = products.dropna(subset=['product_id'])
        products['product_id'] = products['product_id'].astype(int)
        
        departments = pd.read_csv("departments.csv")
    except Exception as e:
        return None, None, None, f"Error loading CSV files: {e}"
    
    # Merge Departments
    if 'department_id' in products.columns and 'department_id' in departments.columns:
        products = products.merge(departments, on="department_id", how="left")
    
    # Load User Item Matrix
    try:
        user_item_matrix = pd.read_csv("user_item_matrix (3) (1).csv", index_col=0)
        # Ensure index (User IDs) and Columns (Product IDs) are normalized
        user_item_matrix.index = user_item_matrix.index.map(safe_int)
        user_item_matrix.columns = [safe_int(c) for c in user_item_matrix.columns]
        
        # Drop invalid columns caused by safe_int returning None
        user_item_matrix = user_item_matrix.loc[:, user_item_matrix.columns.notnull()]
        user_item_matrix = user_item_matrix[user_item_matrix.index.notnull()]
        
    except Exception as e:
        return None, None, None, f"Error loading Matrix: {e}"
    
    # Calculate Item-Item Similarity (Cosine Similarity)
    try:
        # Check if matrix is not empty
        if user_item_matrix.empty:
             return None, None, None, "User-Item Matrix is empty after cleaning."
             
        item_item_matrix = cosine_similarity(user_item_matrix.T)
        item_item_df = pd.DataFrame(item_item_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    except Exception as e:
         return None, None, None, f"Error calculating similarity: {e}"
    
    return products, user_item_matrix, item_item_df, None

# Load SVD Model
@st.cache_resource
def load_model():
    if surprise is None:
        return None
    try:
        with open("recommendation_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None
    except Exception:
        return None

# Load Phase
products_df, user_item_matrix, item_similarity_df, error_msg = load_data()
if error_msg:
    st.error(error_msg)
    st.stop()

svd_model = load_model()

# Initialize Session State
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'just_added' not in st.session_state:
    st.session_state.just_added = None  # Tracks the last product added to trigger recommendation popup

# Recommendation Logic
def get_item_recommendations(product_id, top_n=5):
    """Get similar items based on Item-Item Similarity Matrix, with fallbacks to ensure results."""
    # Ensure ID is consistent (float/int handling)
    recommendations = []
    seen_ids = set()
    
    # 1. Try Content/Collaborative Filtering (Cosine Similarity)
    try:
        pid = float(product_id)
        match = None
        # Try finding as integer/float in index. Our index is Float64 or Int64 often.
        if pid in item_similarity_df.index:
             match = pid
        elif int(pid) in item_similarity_df.index:
             match = int(pid)
        
        if match is not None:
            similar_scores = item_similarity_df[match].sort_values(ascending=False)
            
            # Filter out the item itself (distance 1.0) and any floating point errors near 1.0
            # and ignore items with 0 similarity to avoid random noise unless we want random filling
            
            for idx, score in similar_scores.items():
                # Skip if it is the same item
                if idx == match:
                    continue
                # Skip if score is negligible
                if score < 0.0001:
                    continue
                
                # Add to recs
                rec_id = idx
                recommendations.append((rec_id, score))
                seen_ids.add(rec_id)
                
                if len(recommendations) >= top_n:
                    break
    except Exception as e:
        # print(f"Error in similarity lookup: {e}")
        pass

    # If we have enough, return
    if len(recommendations) >= top_n:
        print(f"DEBUG: ItemRecs found {len(recommendations)} from Similarity Match")
        return recommendations[:top_n]

    # 2. Fallback: Same Department
    try:
        # Find current product's department
        current_prod = products_df[products_df['product_id'] == safe_int(product_id)]
        if not current_prod.empty:
            dept = current_prod.iloc[0]['department']
            # Get other products in same dept
            same_dept_items = products_df[
                (products_df['department'] == dept) & 
                (products_df['product_id'] != safe_int(product_id))
            ]
            
            # Shuffle or sort by popularity if we had that data. Random for now or head.
            # Convert to list
            candidates = same_dept_items['product_id'].tolist()
            
            for cand_id in candidates:
                if len(recommendations) >= top_n:
                    break
                if cand_id not in seen_ids:
                    recommendations.append((cand_id, 0.0)) # 0.0 score for fallback
                    seen_ids.add(cand_id)
            print(f"DEBUG: ItemRecs added Dept Fallback items. Total: {len(recommendations)}")
    except Exception as e:
        pass
        
    # 3. Fallback: Popular / Any Items
    if len(recommendations) < top_n:
        # Just take top items from products df
        all_candidates = products_df['product_id'].tolist()
        for cand_id in all_candidates:
             if len(recommendations) >= top_n:
                break
             if cand_id != safe_int(product_id) and cand_id not in seen_ids:
                 recommendations.append((cand_id, 0.0))
                 seen_ids.add(cand_id)
        print(f"DEBUG: ItemRecs used Global Fallback. Total: {len(recommendations)}")

    return recommendations[:top_n]

def get_user_recommendations(user_id, top_n=5):
    """Get personalized recommendations using SVD Model (if available) or Cart-Based Content Filtering"""
    recommendations = []
    seen_ids = set()
    
    # 1. If User is Logged In & SVD Model Exists -> Use Collaborative Filtering (SVD)
    if svd_model is not None and user_id is not None:
        try:
            # Get items user hasn't rated yet
            if user_id in user_item_matrix.index:
                user_row = user_item_matrix.loc[user_id]
                unseen_items = user_row[user_row == 0].index.tolist()
            else:
                # New user, predict for all
                unseen_items = products_df['product_id'].tolist()
                
            predictions = []
             # Limit unseen items to avoid timeout if list is huge
            cnt = 0
            # Shuffle unseen to not always predict same first 100
            import random
            random_unseen = random.sample(unseen_items, min(len(unseen_items), 200))

            for item_id in random_unseen:
                try:
                    pred = svd_model.predict(uid=user_id, iid=str(item_id))
                    predictions.append((item_id, pred.est))
                except: continue
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            print("DEBUG: UserRecs used SVD.")
            # Return IDs and Source
            return [x[0] for x in predictions[:top_n]], "Personalized (AI)"
        except Exception as e:
            print(f"DEBUG: SVD Error: {e}")
            pass # Fallback
            
    # 2. If User has items in CART -> Use Content/Item-Based Aggregation
    # Find items similar to ANY item in the cart
    if st.session_state.cart:
        print("DEBUG: UserRecs using CART analysis.")
        cart_ids = [safe_int(i) for i in st.session_state.cart if safe_int(i) is not None]
        if cart_ids:
            # Aggregate similarity scores
            scores = {}
            for cid in cart_ids:
                # Get similar items for this cart item
                if cid in item_similarity_df.index:
                    sim_series = item_similarity_df[cid]
                    for pid, score in sim_series.items():
                        if pid == cid: continue
                        if pid in cart_ids: continue # Don't recommend what's already in cart
                        scores[pid] = scores.get(pid, 0) + score
            
            # Sort by total score
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_scores:
                 print(f"DEBUG: Found {len(sorted_scores)} cart-similar items.")
                 return [x[0] for x in sorted_scores[:top_n]], "Based on Cart"

    # 3. Fallback: Popular / Trending (Simple logic: just items with most ratings in matrix, or just random sample)
    # Ensure they are valid products
    print("DEBUG: UserRecs using POPULARITY fallback.")
    if not user_item_matrix.empty:
         # Sum of ratings (if values are ratings) or COUNT of non-zeros
         # Assuming binary or ratings:
         popular_series = (user_item_matrix > 0).sum().sort_values(ascending=False)
         return popular_series.index[:top_n].tolist(), "Popular Items"
         
    # Absolute fallback
    return products_df['product_id'].head(top_n).tolist(), "Generic Top Items"

def add_to_cart(product_id):
    st.session_state.cart.append(str(product_id))
    st.session_state.just_added = str(product_id)
    # Check if we are on home page, if so, we might want to stay there but show notification
    # logic handled in render loop

# --- LOGIN PAGE ---
if st.session_state.user_id is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title("Sign In")
        st.info("Enter your User ID to log in. (Try: 140, 209)")
        
        with st.form("login_form"):
            uid_input = st.text_input("User ID")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if uid_input:
                    val = safe_int(uid_input)
                    if val is not None:
                        st.session_state.user_id = val
                        st.rerun()
                    else:
                        st.error("Please enter a numeric User ID")
                else:
                    st.error("Please enter a User ID")
else:
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown(f"### 👤 User: {st.session_state.user_id}")
        
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.session_state.just_added = None
            st.rerun()
            
        if st.button(f"🛒 Cart ({len(st.session_state.cart)})"):
            st.session_state.page = "cart"
            st.session_state.just_added = None
            st.rerun()
            
        st.markdown("---")
        if st.button("Sign Out"):
            st.session_state.user_id = None
            st.session_state.cart = []
            st.session_state.just_added = None
            st.rerun()

    # --- NOTIFICATION / RECOMMENDATION POPUP logic ---
    if st.session_state.just_added:
        with st.container():
            st.success(f"✅ Added item to cart!")
            st.markdown("### 🔥 Since you added that, you might also like:")
            
            # Show recommendations for the JUST ADDED item using Item-Item (Customers also bought)
            # This is correct for "Immediate cross-sell"
            recs_with_scores = get_item_recommendations(st.session_state.just_added, top_n=4)
            if recs_with_scores:
                cols = st.columns(4)
                for i, (rid, score) in enumerate(recs_with_scores):
                    try:
                         # Safe lookup
                         rid_int = safe_int(rid)
                         item_row = products_df[products_df['product_id'] == rid_int].iloc[0]
                         with cols[i]:
                             st.markdown(f"**{item_row['product_name']}**")
                             st.caption(f"Score: {score:.2f}")
                             if st.button("Add to Cart", key=f"quick_add_{rid}"):
                                 add_to_cart(rid)
                                 st.rerun()
                    except: continue
            st.markdown("---")
            
            if st.button("Close Recommendations"):
                st.session_state.just_added = None
                st.rerun()

    # --- PAGES ---
    if st.session_state.page == "home":
        st.subheader(f"Welcome, User {st.session_state.user_id}")
        
        # Department Filter
        all_depts = products_df['department'].dropna().unique().tolist()
        dept_choice = st.selectbox("Shop by Department", ["All Departments"] + all_depts)
        
        # Filtering
        # 1. Dept Filter
        if dept_choice != "All Departments":
            display_df = products_df[products_df['department'] == dept_choice]
        else:
            display_df = products_df
            
        # 2. MATCHING ONLY Filter (User Request)
        # Only show products that exist in our similarity matrix (so they have recommendations)
        # Intersect product IDs with Matrix Columns
        valid_matrix_ids = set(item_similarity_df.columns)
        display_df = display_df[display_df['product_id'].isin(valid_matrix_ids)]
        
        if display_df.empty:
            st.warning("No products found that match the criteria and have recommendation data.")
        else:
            # Filter for products with at least 1 similar item > 0
            # Pre-calculate validity
            valid_pids_for_display = []
            for pid in display_df['product_id']:
                try:
                    if pid in item_similarity_df.index:
                        # Count similarities > 0 (excluding self)
                        count = (item_similarity_df.loc[pid] > 0.0001).sum() - 1
                        if count >= 1: # User requested 5, but data only supports >=1 for ~8 items. Using 1 to show *something*.
                            valid_pids_for_display.append(pid)
                except: pass
            
            display_df = display_df[display_df['product_id'].isin(valid_pids_for_display)]

            if display_df.empty:
                 st.warning("No products found with sufficient Similarity Data (Similarity > 0).")
            else:
                # Pagination / Limit
                st.write(f"Showing {len(display_df)} products with valid similarities.")
                display_df = display_df.head(40)
            
            # Product Grid
            cols = st.columns(4)
            for idx, row in display_df.iterrows():
                with cols[idx % 4]:
                    st.markdown(f"""
                    <div class="product-card">
                        <div>
                            <div class="product-image-placeholder">
                                <span style="color:#aaa;">Image</span>
                            </div>
                            <div class="product-title" title="{row['product_name']}">{row['product_name']}</div>
                            <div class="product-price">$9.99</div>
                            <div class="prime-badge">✓ prime</div>
                            <div style="font-size: 12px; color: #555;">{str(row['department'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Buttons outside the card div but inside column for Streamlit Layout
                    c1, c2 = st.columns(2)
                    if c1.button("View", key=f"view_{row['product_id']}"):
                        st.session_state.selected_product = row['product_id']
                        st.session_state.page = "product_detail"
                        st.rerun()
                    
                    if c2.button("Add", key=f"add_{row['product_id']}"):
                        add_to_cart(row['product_id'])
                        st.rerun()

    elif st.session_state.page == "product_detail":
        if st.session_state.selected_product is None:
            st.session_state.page = "home"
            st.rerun()
            
        prod_id = st.session_state.selected_product
        try:
            product = products_df[products_df['product_id'] == prod_id].iloc[0]
        except:
            st.error("Product not found")
            st.stop()
            
        if st.button("← Back"):
            st.session_state.page = "home"
            st.rerun()
            
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image("https://via.placeholder.com/400x400?text=Product", use_container_width=True)
        with c2:
            st.title(product['product_name'])
            st.markdown(f"**Department:** {product['department']}")
            st.title("$9.99")
            st.markdown("⭐⭐⭐⭐☆")
            
            if st.button("Add to Cart", key="detail_add", type="primary"):
                add_to_cart(prod_id)
                st.rerun()
        
        # Recommendations Logic
        st.markdown("### Customers who bought this also bought")
        recs_with_scores = get_item_recommendations(prod_id, top_n=5)
        
        if recs_with_scores:
            r_cols = st.columns(5)
            for i, (rid, score) in enumerate(recs_with_scores):
                 try:
                     rid_int = safe_int(rid)
                     item = products_df[products_df['product_id'] == rid_int].iloc[0]
                     with r_cols[i]:
                         st.markdown(f"**{item['product_name']}**")
                         st.caption(f"Similarity: {score:.2f}")
                         st.button("View", key=f"rec_view_{rid}", on_click=lambda r=rid_int: setattr(st.session_state, 'selected_product', r))
                 except: pass
        else:
            st.write("No specific recommendations found for this item.")
        st.title("Shopping Cart")
        
        if not st.session_state.cart:
            st.info("Your cart is empty.")
        else:
            cart_ids = st.session_state.cart
            
            # List items
            for item_id in cart_ids:
                try:
                    rid_int = safe_int(item_id)
                    item = products_df[products_df['product_id'] == rid_int].iloc[0]
                    with st.container():
                        st.markdown(f"**{item['product_name']}** - $9.99")
                except: pass
                
            st.markdown("---")
            st.button("Proceed to Checkout", type="primary")
            
            # Recommendation engine for Cart
            st.markdown("### Recommended based on your cart")
            if cart_ids:
                # Use the last added item for recs
                last = cart_ids[-1]
                # get_item_recommendations returns [(id, score), ...]
                cart_recs_with_scores = get_item_recommendations(last, top_n=4)
                
                # Extract just IDs for display (or use scores if needed)
                # Here we just want the list of valid recommendations
                
                c_cols = st.columns(4)
                for i, (rid, score) in enumerate(cart_recs_with_scores):
                    try:
                        rid_int = safe_int(rid)
                        item = products_df[products_df['product_id'] == rid_int].iloc[0]
                        with c_cols[i]:
                            st.caption(item['product_name'])
                            st.caption(f"Match: {score:.2f}")
                            if st.button("Add", key=f"cart_add_{rid}"):
                                add_to_cart(rid)
                                st.rerun()
                    except: pass


