"""
Collaborative Filtering with ALS + FAISS
----------------------------------------
This script trains an implicit Alternating Least Squares (ALS) model
on buyer-lot bid data, extracts embeddings, builds a FAISS index for
efficient similarity search, and generates lot recommendations for buyers.
"""

# ==============================================================
# Imports
# ==============================================================
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import os


# ==============================================================
# 1️⃣ Encoder & Matrix Builders
# ==============================================================
def build_encoders(data: pd.DataFrame):
    """Fit label encoders for buyer and lot IDs."""
    buyer_encoder = LabelEncoder()
    lot_encoder = LabelEncoder()

    buyer_ids = buyer_encoder.fit_transform(data['buyer_nbr'])
    lot_ids = lot_encoder.fit_transform(data['lot_nbr'])

    return buyer_encoder, lot_encoder, buyer_ids, lot_ids


def build_sparse_matrix(data: pd.DataFrame, buyer_ids, lot_ids):
    """Create a buyer-lot sparse matrix weighted by max_bid."""
    max_bid_values = data['max_bid'].fillna(0).astype(float)
    n_buyers = len(np.unique(buyer_ids))
    n_lots = len(np.unique(lot_ids))

    sparse_matrix = csr_matrix((max_bid_values, (buyer_ids, lot_ids)), shape=(n_buyers, n_lots))
    return sparse_matrix


# ==============================================================
# 2️⃣ ALS Model Training & Embedding Extraction
# ==============================================================
def train_als_model(sparse_matrix, factors=32, regularization=0.5, iterations=30, use_gpu=False):
    """Train implicit ALS model."""
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=use_gpu
    )
    model.fit(sparse_matrix)
    return model


def extract_embeddings(als_model):
    """Extract buyer and lot embeddings from the trained ALS model."""
    buyer_embeddings = als_model.user_factors.astype('float32')
    lot_embeddings = als_model.item_factors.astype('float32')
    return buyer_embeddings, lot_embeddings


# ==============================================================
# 3️⃣ FAISS Index
# ==============================================================
def build_faiss_index(buyer_embeddings):
    """Build FAISS index using normalized buyer embeddings."""
    faiss.normalize_L2(buyer_embeddings)
    dim = buyer_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(buyer_embeddings)
    return index


# ==============================================================
# 4️⃣ Recommendation Functions
# ==============================================================
def get_similar_buyers_faiss(input_buyer_id, buyer_encoder, buyer_embeddings, faiss_index, als_model,top_k=5):
    """Return top-k similar buyers for a given buyer using FAISS."""
    if input_buyer_id not in buyer_encoder.classes_:
        raise ValueError("Buyer not in training data")

    internal_buyer_id = buyer_encoder.transform([input_buyer_id])[0]

    # Get query embedding and normalize
    query_vec = als_model.user_factors[internal_buyer_id].astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vec)

    distances, indices = faiss_index.search(query_vec, top_k + 1)
    similar_ids = indices[0]
    similar_ids = [i for i in similar_ids if i != internal_buyer_id][:top_k]
    similar_buyers = buyer_encoder.inverse_transform(similar_ids)

    return similar_buyers



def recommend_lots_cosine_from_similar_buyers(input_buyer_id, data, buyer_encoder, lot_encoder, buyer_embeddings, lot_embeddings,
    als_model, faiss_index, top_k_buyers=5, top_k_lots=6):
    """Generate lot recommendations for a buyer based on similar buyers' behavior."""
    # Step 1: Find similar buyers
    similar_buyers = get_similar_buyers_faiss(
        input_buyer_id, buyer_encoder, buyer_embeddings, faiss_index, als_model, top_k=top_k_buyers
    )

    # Step 2: Get lots interacted by similar buyers
    sim_buyer_lots = data[data['buyer_nbr'].isin(similar_buyers)]
    candidate_lot_ids = sim_buyer_lots['lot_nbr'].unique()

    # Step 3: Remove already seen lots
    input_buyer_lot_ids = data[data['buyer_nbr'] == input_buyer_id]['lot_nbr'].unique()
    candidate_lot_ids = list(set(candidate_lot_ids) - set(input_buyer_lot_ids))
    if not candidate_lot_ids:
        return pd.DataFrame()

    # Step 4: Get buyer and lot embeddings
    input_buyer_idx = buyer_encoder.transform([input_buyer_id])[0]
    buyer_vec = buyer_embeddings[input_buyer_idx].reshape(1, -1)
    lot_indices = lot_encoder.transform(candidate_lot_ids)
    lot_vecs = lot_embeddings[lot_indices]
    faiss.normalize_L2(lot_vecs)

    # Step 5: Cosine similarity = dot product of normalized vectors
    cosine_scores = np.dot(lot_vecs, buyer_vec.T).flatten()

    # Step 6: Top lots
    top_indices = np.argsort(-cosine_scores)[:top_k_lots]
    top_lot_ids = [candidate_lot_ids[i] for i in top_indices]
    top_scores = cosine_scores[top_indices]

    # Step 7: Build recommendation DataFrame
    top_rows = []
    for lot_id, score in zip(top_lot_ids, top_scores):
        matching_rows = sim_buyer_lots[(sim_buyer_lots['lot_nbr'] == lot_id) & (sim_buyer_lots['buyer_nbr'].isin(similar_buyers))]

        if matching_rows.empty:
            continue  # skip this lot if no matching similar buyer row found

        row = matching_rows.iloc[0]

        top_rows.append({
            'input_buyer_nbr': input_buyer_id,
            'mbr_email': row['mbr_email'],
            'recommended_lot': lot_id,
            'lot_year': row['lot_year'],
            'lot_make_cd': row['lot_make_cd'],
            'grp_model': row['grp_model'],
            'acv': row['acv'],
            'repair_cost': row['repair_cost'],
            'inv_dt': row['inv_dt'],
            'cosine_similarity': score
        })

    return pd.DataFrame(top_rows)


# ==============================================================
# 5️⃣ Batch Runner
# ==============================================================

def run_batch_recommendations(data, output_path: str):

    print("\nBuilding encoders and sparse matrix...")
    buyer_encoder, lot_encoder, buyer_ids, lot_ids = build_encoders(data)
    sparse_matrix = build_sparse_matrix(data, buyer_ids, lot_ids)

    print("Training ALS model...")
    als_model = train_als_model(sparse_matrix)

    print("Extracting embeddings and building FAISS index...")
    buyer_embeddings, lot_embeddings = extract_embeddings(als_model)
    faiss_index = build_faiss_index(buyer_embeddings)

    print("Generating recommendations...")
    all_buyers = data['buyer_nbr'].unique()
    all_recos = []

    for buyer in tqdm(all_buyers):
        try:
            df = recommend_lots_cosine_from_similar_buyers(
                input_buyer_id=buyer,
                data=data,
                buyer_encoder=buyer_encoder,
                lot_encoder=lot_encoder,
                buyer_embeddings=buyer_embeddings,
                lot_embeddings=lot_embeddings,
                als_model=als_model,
                faiss_index=faiss_index
            )
            if not df.empty:
                all_recos.append(df)
        except Exception as e:
            print(f"⚠️ Error for buyer {buyer}: {e}")

    recommendations_df = pd.concat(all_recos, ignore_index=True)

    save_processed_data(recommendations_df, output_path)
    return recommendations_df, buyer_encoder, buyer_embeddings, faiss_index, als_model


# ==============================================================
# 6️⃣ Save to Excel Helper (Always Excel)
# ==============================================================

def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save a DataFrame to Excel (.xlsx) in a given path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    print(f"\n✅ Saved processed data to {output_path} ({len(df):,} rows)")


# ==============================================================
# 6️⃣ Entry Point
# ==============================================================
if __name__ == "__main__":
    cf_test = pd.read_csv("data/split/cf_test.csv")
    cf_holdout = pd.read_csv("data/split/cf_holdout.csv")

    run_batch_recommendations(cf_test, output_path = "data/past_reco/cf_test_reco.xlsx")
    run_batch_recommendations(cf_holdout, output_path = "data/past_reco/cf_holdout_would_have_reco.xlsx" )
