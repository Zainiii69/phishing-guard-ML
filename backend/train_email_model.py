import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Paths
DATASET_PATH = os.path.join(os.path.dirname(__file__), '../dataset/Phishing_Email.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml/email_phishing_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '../ml/email_vectorizer.pkl')

def train_model():
    print("Loading email dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(df)} rows.")
    
    # 1. Preprocessing & Cleaning
    # Drop rows with null values
    df = df.dropna(subset=['Email Text', 'Email Type'])
    print(f"Cleaned dataset: {len(df)} rows after removing nulls.")
    
    # Map labels: Phishing Email -> 1, Safe Email -> 0
    # Note: Kaggle dataset uses 'Phishing Email' and 'Safe Email'
    df['label'] = df['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0})
    
    # Check if mapping succeeded
    if df['label'].isnull().any():
        print("Warning: Some labels could not be mapped correctly. Unique values in Email Type:")
        print(df['Email Type'].unique())
        # Drop any remaining unmapped rows
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

    X = df['Email Text']
    y = df['label']
    
    print("\nLabel distribution:")
    print(y.value_counts(normalize=True))
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {len(X_train)} samples, Testing set: {len(X_test)} samples.")
    
    # 3. Vectorization (TF-IDF)
    print("\nVectorizing text data (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 4. Model Training
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_vec, y_train)
    
    # 5. Evaluation
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe Email', 'Phishing Email']))
    
    # 6. Save Model and Vectorizer
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Saving vectorizer to {VECTORIZER_PATH}...")
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Email model training complete successfully!")

if __name__ == "__main__":
    train_model()
