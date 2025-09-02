from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- Simulating Model and Preprocessor Loading ---
# In a real-world application, you would load your .pkl files like this:
# with open('preprocessor.pkl', 'rb') as f:
#     preprocessor = pickle.load(f)
# with open('kmeans_model.pkl', 'rb') as f:
#     kmeans_model = pickle.load(f)

# Hardcoded data from the previous analysis to simulate the loaded files.
# This ensures the code is runnable without the original .pkl files.
SCALER_PARAMS = {
    'mean': np.array([0.370321, 104.148107, 1.854944, 0.103091, 0.005085, 0.221516, 101.996135, 0.570146, 3.424419]),
    'scale': np.array([0.482937, 106.879737, 0.575608, 0.398696, 0.072230, 0.651765, 47.973302, 0.793854, 2.404287])
}

# Cluster centers representing the four segments
CLUSTER_CENTERS = np.array([
    [0.162410, 73.774288, 1.838520, 0.126588, 0.008453, 0.137837, 86.914285, 0.673886, 3.469428],
    [0.238496, 75.092213, 1.860601, 0.134882, 0.009491, 0.155380, 96.115201, 0.727441, 3.567476],
    [0.182713, 77.617068, 2.113063, 0.222277, 0.009943, 0.211475, 121.285810, 1.531729, 4.158643],
    [0.926578, 206.007654, 1.944299, 0.040187, 0.000371, 0.015242, 78.490892, 0.082599, 3.064170]
])

CLUSTER_DESCRIPTIONS = [
    {"name": "The Dependable Planners", "summary": "Organized, low-risk, and have solid plans.", "emoji": "🗓️"},
    {"name": "The Flexible Online Shoppers", "summary": "Price-sensitive and more likely to cancel.", "emoji": "🛒"},
    {"name": "The Family Vacationers", "summary": "Book for longer stays and require more amenities.", "emoji": "👨‍👩‍👧‍👦"},
    {"name": "The High-Risk Group Bookers", "summary": "Large group bookings with very high cancellation rates.", "emoji": "⚠️"}
]

# Root route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Prepare the input data as a NumPy array
        # Ensure the order of features matches the model's training data
        features = [
            int(data['is_canceled']),
            float(data['lead_time']),
            float(data['adults']),
            float(data['children']),
            float(data['babies']),
            float(data['booking_changes']),
            float(data['adr']),
            float(data['total_of_special_requests']),
            float(data['total_nights']),
        ]
        
        input_array = np.array(features).reshape(1, -1)
        
        # Scale the input data using the hardcoded scaler parameters
        scaled_input = (input_array - SCALER_PARAMS['mean']) / SCALER_PARAMS['scale']
        
        # Find the closest cluster center using Euclidean distance
        distances = np.linalg.norm(scaled_input - CLUSTER_CENTERS, axis=1)
        cluster_id = np.argmin(distances)

        # Get the description for the predicted cluster
        predicted_cluster_info = CLUSTER_DESCRIPTIONS[cluster_id]

        # Return the prediction as a JSON response
        return jsonify(predicted_cluster_info)

    except Exception as e:
        # Log the error and return an error message
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction. Please check your input."}), 400

if __name__ == '__main__':
    # You can run this app from the command line using 'python app.py'
    # Use debug=True for development, turn off for production
    app.run(debug=True)
