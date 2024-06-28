import tensorflow as tf
import json
import joblib
import pickle
import os

# Function to load Twitter data from JSON file
def load_twitter_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Function to preprocess texts for prediction
def preprocess_texts(texts, tokenizer, max_len=80):
    if not texts:  # Handle empty input case
        raise ValueError("Empty list of texts provided")

    tokens = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    return input_ids, attention_mask

# Function to predict emotions using BERT model
def predict_emotions(texts, tokenizer, bert_model, predefined_emotions):
    input_ids, attention_mask = preprocess_texts(texts, tokenizer)
    if input_ids is None or attention_mask is None:
        raise ValueError("Failed to preprocess texts")

    # Predict using the BERT model loaded
    inputs = [input_ids, attention_mask]
    predictions = bert_model(inputs, training=False)

    # Process predictions
    predicted_labels = [predefined_emotions[idx] for idx in tf.argmax(predictions, axis=1)]
    return predicted_labels

# Sample usage to load JSON data and predict emotions
def main():
    # Load Tokenizer using joblib
    tokenizer_file = 'tokenizer_bert_final.pkl'
    tokenizer = joblib.load(tokenizer_file)

    # Load BERT model using tf.saved_model.load
    model_dir = 'bert_emotion final'
    try:
        bert_model = tf.saved_model.load(model_dir)
        print("BERT Model loaded successfully!")
    except OSError as e:
        print(f"Error loading the BERT model: {e}")
        return

    # Load predefined emotions
    with open('emotion_classes.txt', 'r') as file:
        predefined_emotions = [line.strip() for line in file.readlines()]

    # Load JSON data from file (replace with your JSON file path)
    json_file = (r'C:\Users\adria\Documents\GitHub\Sentiment-Analysis\WebAppS\web_scrapping_apify.json') # Example of absolute path
    if not os.path.isfile(json_file):
        print(f"Error: JSON file '{json_file}' not found.")
        return

    try:
        twitter_data = load_twitter_data(json_file)

        # Extract texts from Twitter data
        texts = [tweet['text'] for tweet in twitter_data]

        # Predict emotions
        predicted_emotions = predict_emotions(texts, tokenizer, bert_model, predefined_emotions)

        # Print or process predicted emotions
        for tweet, emotion in zip(twitter_data, predicted_emotions):
            print(f"Tweet: {tweet['text']}")
            print(f"Predicted Emotion: {emotion}")
            print("-" * 10)

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
