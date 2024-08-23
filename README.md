# Intent Classification API

This API provides endpoints for classifying intents based on user utterances using a pre-trained machine learning model.

## Table of Contents

1. [Setup](#setup)
2. [API Endpoints](#api-endpoints)
   - [Classify Intent](#classify-intent)
   - [Classify Batch](#classify-batch)
   - [Health Check](#health-check)
3. [Usage Examples](#usage-examples)

## Setup

1. Ensure you have Python installed on your system.
2. Install the required dependencies:
   ```
   pip install flask scikit-learn joblib
   ```
3. Make sure you have the trained model file `intent_classifier.joblib` in the same directory as your script.
4. Run the Flask application:
   ```
   python app.py
   ```
   The API will be available at `http://localhost:5000`.

## API Endpoints

### Classify Intent

Classifies the intent of a single utterance.

- **URL:** `/classify_intent`
- **Method:** POST
- **Request Body:**
  ```json
  {
    "utterance": "What's the weather like today?"
  }
  ```
- **Response:**
  ```json
  {
    "utterance": "What's the weather like today?",
    "intent": "get_weather",
    "confidence": 0.95
  }
  ```

### Classify Batch

Classifies the intents of multiple utterances in a single request.

- **URL:** `/classify_batch`
- **Method:** POST
- **Request Body:**
  ```json
  {
    "utterances": [
      "What's the weather like today?",
      "Set an alarm for 7 AM",
      "How tall is Mount Everest?"
    ]
  }
  ```
- **Response:**
  ```json
  [
    {
      "utterance": "What's the weather like today?",
      "intent": "get_weather",
      "confidence": 0.95
    },
    {
      "utterance": "Set an alarm for 7 AM",
      "intent": "set_alarm",
      "confidence": 0.88
    },
    {
      "utterance": "How tall is Mount Everest?",
      "intent": "general_query",
      "confidence": 0.72
    }
  ]
  ```

### Health Check

Checks if the API is running and healthy.

- **URL:** `/health`
- **Method:** GET
- **Response:**
  ```json
  {
    "status": "healthy"
  }
  ```

## Usage Examples

Here are some examples of how to use the API with curl:

1. Classify a single intent:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"utterance": "What's the weather like today?"}' http://localhost:5000/classify_intent
   ```

2. Classify multiple intents:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"utterances": ["What's the weather like today?", "Set an alarm for 7 AM"]}' http://localhost:5000/classify_batch
   ```

3. Check API health:
   ```
   curl http://localhost:5000/health
   ```
