from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained sentiment-analysis model from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis")

# Route to handle sentiment analysis
@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    # Get the JSON data from the request
    data = request.get_json()

    # Check if the 'text' field is in the received data
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    # Extract the text from the data
    text = data['text']
    
    # Perform sentiment analysis
    result = sentiment_analyzer(text)
    
    # Return the sentiment label and score
    return jsonify(result[0])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
