import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import nltk

# Set NLTK data path
nltk.data.path.append('/usr/local/share/nltk_data')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    try:
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Join tokens back into string
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text  # Return original text if tokenization fails

class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.label_encoder = None
        self.confidence_threshold = 0.35  # Threshold for out-of-scope detection

    def train(self, X, y):
        # Preprocess the text data
        X = [preprocess_text(text) for text in X]
        
        # Encode labels
        self.label_encoder = {label: i for i, label in enumerate(set(y))}
        y_encoded = [self.label_encoder[label] for label in y]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.pipeline.predict(X_test)
        print(classification_report([list(self.label_encoder.keys())[i] for i in y_test], 
                                    [list(self.label_encoder.keys())[i] for i in y_pred]))

    def predict(self, texts):
        # Preprocess the input texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Get predictions and probabilities
        predictions = self.pipeline.predict(processed_texts)
        probabilities = self.pipeline.predict_proba(processed_texts)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            confidence = np.max(prob)
            if confidence < self.confidence_threshold:
                intent = "out_of_scope"
            else:
                intent = list(self.label_encoder.keys())[pred]
            results.append((intent, float(confidence)))
        
        return results

    def save(self, path):
        joblib.dump({
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'confidence_threshold': self.confidence_threshold
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        classifier = cls()
        classifier.pipeline = data['pipeline']
        classifier.label_encoder = data['label_encoder']
        classifier.confidence_threshold = data['confidence_threshold']
        return classifier

# Example usage
if __name__ == '__main__':
    # Sample data (replace with your actual data)
    data = pd.read_csv('data.csv')

    classifier = IntentClassifier()
    classifier.train(data['text'], data['intent'])
    classifier.save('intent_classifier.joblib')

    # Test prediction
    test_texts = [
        "What's the temperature outside?",
        "Can you set my alarm clock?",
        "I want to listen to some songs",
        "Is there anything on my schedule today?",
        "Tell me something funny",
        "Who was the first president of the United States?",
        "Can you turn off the kitchen lights?",
        "What's the square root of 144?",
    ]
    
    loaded_classifier = IntentClassifier.load('intent_classifier.joblib')
    predictions = loaded_classifier.predict(test_texts)
    
    for text, (intent, confidence) in zip(test_texts, predictions):
        print(f"Text: {text}")
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.2f}")
        print()