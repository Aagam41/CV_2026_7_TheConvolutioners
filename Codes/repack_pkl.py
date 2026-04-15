import pickle

with open('rf_sift.pkl', 'rb') as f:
    rf = pickle.load(f)
with open('vocab_sift.pkl', 'rb') as f:
    kmeans = pickle.load(f)

blob = {
    'model': rf,
    'kmeans': kmeans,
    'n_clusters': kmeans.n_clusters,   # 100 in the Kaggle script
    'sift_nfeatures': 100,
    'crop_size': (64, 64),
    'feature_type': 'sift_bow',
}
with open('rf_sift_bundled.pkl', 'wb') as f:
    pickle.dump(blob, f)
print('Wrote rf_sift_bundled.pkl')
