from flask import Flask, request, jsonify
from intent_classifier import IntentClassifier, preprocess_text

app = Flask(__name__)

# Load the trained model
classifier = IntentClassifier.load('intent_classifier.joblib')

@app.route('/classify_intent', methods=['POST'])
def classify_intent():
    data = request.json
    if 'utterance' not in data:
        return jsonify({'error': 'No utterance provided'}), 400
    
    utterance = data['utterance']
    intent, confidence = classifier.predict([utterance])[0]
    
    return jsonify({
        'utterance': utterance,
        'intent': intent,
        'confidence': confidence
    })

@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    data = request.json
    if 'utterances' not in data or not isinstance(data['utterances'], list):
        return jsonify({'error': 'Invalid or missing utterances'}), 400
    
    utterances = data['utterances']
    predictions = classifier.predict(utterances)
    
    results = [
        {'utterance': utterance, 'intent': intent, 'confidence': confidence}
        for utterance, (intent, confidence) in zip(utterances, predictions)
    ]
    
    return jsonify(results)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)