import os
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from python.fraud_dataset import FRAUD_DATA

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_model.joblib')

class FraudModelManager:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._initialize_model()

    def _preprocess_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def _initialize_model(self):
        """Train or load the model"""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print("[ML Model] Loaded existing model from disk.")
                return
            except Exception as e:
                print(f"[ML Model] Error loading model: {e}. Retraining...")

        self._train_new_model()

    def _train_new_model(self):
        """Train a fresh model from the dataset"""
        print("[ML Model] Training new model with dataset...")
        
        texts = [self._preprocess_text(item[0]) for item in FRAUD_DATA]
        labels = [item[1] for item in FRAUD_DATA]

        # Create a pipeline with TF-IDF Vectorizer and Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
            ('clf', MultinomialNB(alpha=0.1))
        ])

        self.model.fit(texts, labels)
        
        # Save the model
        try:
            # Try to save, but don't fail if read-only
            if not os.environ.get('VERCEL'):
                joblib.dump(self.model, MODEL_PATH)
                print("[ML Model] Model trained and saved to disk.")
            else:
                print("[ML Model] Model trained but not saved (Vercel read-only).")
        except Exception as e:
            print(f"[ML Model] Could not save model: {e}")

    def predict(self, message):
        """Predict if a message is fraud and return score/label"""
        if not self.model:
            return 0.0, "unknown", 0.0

        clean_message = self._preprocess_text(message)
        
        # Get probability
        # [0, 1] -> prob for each class
        try:
            probs = self.model.predict_proba([clean_message])[0]
            fraud_prob = float(probs[1])
            
            # Decide label
            if fraud_prob >= 0.7:
                label = "fraud"
                confidence = fraud_prob
            elif fraud_prob >= 0.4:
                label = "suspicious"
                confidence = 0.5 + (fraud_prob - 0.4) / 0.3 * 0.3
            else:
                label = "safe"
                confidence = 1.0 - fraud_prob
                
            return fraud_prob * 100, label, confidence
        except Exception as e:
            print(f"[ML Model] Prediction error: {e}")
            return 0.0, "error", 0.0

# Singleton instance
model_manager = FraudModelManager()

if __name__ == "__main__":
    # Test
    test_msgs = [
        "Congratulations you won a lottery!",
        "Hey how are you doing today?",
        "Your bank account is suspended, click here now!"
    ]
    for m in test_msgs:
        score, label, conf = model_manager.predict(m)
        print(f"Msg: {m} -> Score: {score:.1f}, Label: {label}, Conf: {conf:.2f}")
