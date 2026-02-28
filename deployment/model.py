import pickle
import gzip
import pandas as pd
import numpy as np

with open('models/logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

with gzip.open("models/user_final_rating.pkl.gz", "rb") as f:
    user_final_rating = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

sentiment_df = pd.read_csv('data/sentiment_df.csv')

# reviews = pd.read_csv('data/sentiment_df.csv')

def recommend_products_hybrid(user_input, cf_weight=0.4, sentiment_weight=0.6):

    if user_input not in user_final_rating.index:
        raise KeyError(f"User {user_input} not found")

    user_cf = user_final_rating.loc[user_input]
    user_cf_nonzero = user_cf[user_cf > 0].sort_values(ascending=False)

    if user_cf_nonzero.empty:
        return []

    product_sentiment = {}
    items_to_process = user_cf_nonzero.head(20).index.tolist()

    max_cf = user_cf_nonzero.max()

    for item_id in items_to_process:
        item_reviews = sentiment_df[sentiment_df['id'] == item_id]

        if item_reviews.empty:
            product_sentiment[item_id] = 0.5
            continue

        item_lemmatized_reviews = item_reviews['reviews_lemmatized'].astype(str)
        item_tfidf_vectors = tfidf_vectorizer.transform(item_lemmatized_reviews)
        preds = model.predict(item_tfidf_vectors)

        product_sentiment[item_id] = np.mean(preds) if len(preds) > 0 else 0.5

    hybrid_scores = []

    for item_id in items_to_process:
        cf_rating = user_cf_nonzero[item_id]
        sentiment_score = product_sentiment.get(item_id, 0.5)

        cf_norm = cf_rating / max_cf if max_cf > 0 else 0
        hybrid_score = cf_weight * cf_norm + sentiment_weight * sentiment_score

        hybrid_scores.append({
            'id': item_id,
            'hybrid_score': hybrid_score
        })

    d_hybrid = pd.DataFrame(hybrid_scores)
    d_hybrid = d_hybrid.sort_values('hybrid_score', ascending=False).head(5)

    d_hybrid = pd.merge(
        d_hybrid,
        sentiment_df[['id', 'name']].drop_duplicates(),
        on='id',
        how='left'
    )

    return d_hybrid['name'].dropna().tolist()


if __name__ == "__main__":
    user_id = input("Enter your user ID: ")
    print(recommend_products_hybrid(user_id))