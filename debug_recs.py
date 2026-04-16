import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Mock Session State
class MockSessionState:
    def __init__(self):
        self.cart = []
        self.user_id = None
        self.just_added = None

if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# --- COPY PASTE LOGIC FROM APP (Modified to run standalone) ---
def safe_int(x):
    try:
        return int(float(x))
    except:
        return None

print("Loading Data...")
products_df = pd.read_csv("products.csv")
products_df['product_id'] = products_df['product_id'].apply(safe_int)
products_df = products_df.dropna(subset=['product_id'])
products_df['product_id'] = products_df['product_id'].astype(int)

departments = pd.read_csv("departments.csv")
if 'department_id' in products_df.columns and 'department_id' in departments.columns:
    products_df = products_df.merge(departments, on="department_id", how="left")

user_item_matrix = pd.read_csv("user_item_matrix (3) (1).csv", index_col=0)
user_item_matrix.index = user_item_matrix.index.map(safe_int)
user_item_matrix.columns = [safe_int(c) for c in user_item_matrix.columns]
user_item_matrix = user_item_matrix.loc[:, user_item_matrix.columns.notnull()]
user_item_matrix = user_item_matrix[user_item_matrix.index.notnull()]

item_item_matrix = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_item_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

svd_model = None
try:
    with open("recommendation_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
except: pass

print("Data Loaded.")

# --- FUNCTIONS UNDER TEST ---
def get_user_recommendations(user_id, top_n=5):
    recommendations = []
    seen_ids = set()
    
    # 1. SVD
    if svd_model is not None and user_id is not None:
        try:
            if user_id in user_item_matrix.index:
                user_row = user_item_matrix.loc[user_id]
                unseen_items = user_row[user_row == 0].index.tolist()
            else:
                unseen_items = products_df['product_id'].tolist()
            
            predictions = []
            import random
            random_unseen = random.sample(unseen_items, min(len(unseen_items), 200))

            for item_id in random_unseen:
                try:
                    pred = svd_model.predict(uid=user_id, iid=str(item_id))
                    predictions.append((item_id, pred.est))
                except: continue
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            print("DEBUG: UserRecs used SVD.")
            return [x[0] for x in predictions[:top_n]]
        except Exception as e:
            print(f"DEBUG: SVD Error: {e}")
            pass 

    # 2. Cart
    if st.session_state.cart:
        print("DEBUG: UserRecs using CART analysis.")
        cart_ids = [safe_int(i) for i in st.session_state.cart if safe_int(i) is not None]
        if cart_ids:
            scores = {}
            for cid in cart_ids:
                if cid in item_similarity_df.index:
                    sim_series = item_similarity_df[cid]
                    for pid, score in sim_series.items():
                        if pid == cid: continue
                        if pid in cart_ids: continue 
                        scores[pid] = scores.get(pid, 0) + score
            
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_scores:
                 print(f"DEBUG: Found {len(sorted_scores)} cart-similar items.")
                 return [x[0] for x in sorted_scores[:top_n]]

    # 3. Popularity
    print("DEBUG: UserRecs using POPULARITY fallback.")
    if not user_item_matrix.empty:
         popular_series = (user_item_matrix > 0).sum().sort_values(ascending=False)
         return popular_series.index[:top_n].tolist()
         
    return products_df['product_id'].head(top_n).tolist()

def get_item_recommendations(product_id, top_n=5):
    recommendations = []
    seen_ids = set()
    
    # 1. Similarity
    try:
        pid = float(product_id)
        match = None
        if pid in item_similarity_df.index: match = pid
        elif int(pid) in item_similarity_df.index: match = int(pid)
        
        if match is not None:
            similar_scores = item_similarity_df[match].sort_values(ascending=False)
            for idx, score in similar_scores.items():
                if idx == match: continue
                if score < 0.0001: continue
                recommendations.append((idx, score))
                seen_ids.add(idx)
                if len(recommendations) >= top_n: break
            if len(recommendations) >= top_n:
                print(f"DEBUG: ItemRecs found {len(recommendations)} from Similarity Match")
                return recommendations[:top_n]
    except: pass

    # 2. Dept
    try:
        current_prod = products_df[products_df['product_id'] == safe_int(product_id)]
        if not current_prod.empty:
            dept = current_prod.iloc[0]['department']
            same_dept_items = products_df[(products_df['department'] == dept) & (products_df['product_id'] != safe_int(product_id))]
            candidates = same_dept_items['product_id'].tolist()
            for cand_id in candidates:
                if len(recommendations) >= top_n: break
                if cand_id not in seen_ids:
                    recommendations.append((cand_id, 0.0))
                    seen_ids.add(cand_id)
            print(f"DEBUG: ItemRecs added Dept Fallback items. Total: {len(recommendations)}")
    except: pass
        
    # 3. Global
    if len(recommendations) < top_n:
        all_candidates = products_df['product_id'].tolist()
        for cand_id in all_candidates:
             if len(recommendations) >= top_n: break
             if cand_id != safe_int(product_id) and cand_id not in seen_ids:
                 recommendations.append((cand_id, 0.0))
                 seen_ids.add(cand_id)
        print(f"DEBUG: ItemRecs used Global Fallback. Total: {len(recommendations)}")

    return recommendations[:top_n]

print("\n--- TEST 1: GUEST USER (No ID, Empty Cart) ---")
urecs = get_user_recommendations(None)
print(f"User Recs: {urecs}")

print("\n--- TEST 2: ITEM REC (Product with low similarity to force fallback) ---")
# Pick a product
pid = products_df['product_id'].iloc[0]
irecs = get_item_recommendations(pid)
print(f"Item Recs for {pid}: {[r[0] for r in irecs]}")

print("\n--- TEST 3: GUEST USER (Cart has item) ---")
st.session_state.cart = [str(pid)]
urecs_cart = get_user_recommendations(None)
print(f"User Recs (Cart={pid}): {urecs_cart}")

print("\n--- COMPARISON ---")
print(f"UserRecs (Empty): {urecs}")
print(f"ItemRecs ({pid}): {[r[0] for r in irecs]}")
if set(urecs) == set([r[0] for r in irecs]):
    print("WARNING: Recs are IDENTICAL!")
else:
    print("SUCCESS: Recs are DIFFERENT.")
