
# Example of how to load and use the saved model for predictions
import pickle
import numpy as np
from scipy.sparse import csr_matrix

# Load the trained model
with open('models/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the feature encoders
with open('features/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
with open('features/target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

# Example: Make predictions on new data
# Assuming you have new text data to vectorize
# new_text_features = ["sample video title and description"]
# X_new = tfidf_vectorizer.transform(new_text_features)

# For sparse matrix input (like our processed features):
# X_new should be in the same format as our training features
# predictions = model.predict(X_new)
# predicted_categories = target_encoder.inverse_transform(predictions)

print("Model loaded and ready for predictions!")
