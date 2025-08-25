from flask import Flask, request, jsonify
import pandas as pd
import sys
import os

# Get the absolute path of the current working directory.
# This should be the project's root, as you are running the script from there.
project_root = os.getcwd()

# Add the project's root directory to the system path.
# This is the most reliable way to ensure Python can find the 'src' package.
if project_root not in sys.path:
    sys.path.append(project_root)

# Verify the updated path for debugging purposes
print("Updated sys.path:", sys.path)

# Now, we can import our custom classes
from src.exception import CustomException
from src.pipelines.predict_pipeline import PredictPipeline

# Create a Flask web application instance
app = Flask(__name__)

# --- Load the Prediction Pipeline globally ---
# This is crucial for efficiency. The model and preprocessor are loaded only once
# when the app starts, not on every request.
try:
    predict_pipeline = PredictPipeline()
    print("Prediction pipeline loaded successfully.")
except Exception as e:
    # If there is an issue loading the pipeline, the app should not run
    print(f"Error loading prediction pipeline: {e}", file=sys.stderr)
    sys.exit(1)

# --- Define the API routes ---

@app.route('/')
def home():
    """
    A simple route to check if the API is running.
    """
    return jsonify({"message": "Hotel Customer Segmentation API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    An API endpoint to receive customer data and return their cluster segment.
    """
    try:
        # Get the JSON data from the POST request
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received."}), 400

        # Convert the JSON data into a Pandas DataFrame
        df = pd.DataFrame(data, index=[0])

        # Call the predict method from our pipeline class
        # It handles data cleaning, transformation, and prediction
        segmented_df = predict_pipeline.predict(df)
        
        # Extract the cluster and the cleaned customer data for the response
        cluster_assignment = segmented_df['cluster'].iloc[0]
        
        # Convert the entire segmented DataFrame to a dictionary
        response_data = segmented_df.to_dict(orient='records')[0]

        # Return the response as JSON
        return jsonify({
            "status": "success",
            "message": "Prediction successful.",
            "customer_data": response_data,
            "assigned_cluster": int(cluster_assignment)
        })

    except CustomException as e:
        # Handle custom exceptions raised by our pipeline
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    except Exception as e:
        # Handle any other unexpected errors
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred.",
            "details": str(e)
        }), 500

# --- Main execution block to run the app ---
if __name__ == '__main__':
    # Start the Flask development server
    # The debug=True flag is helpful for development, but should be False in production
    app.run(debug=True, host='0.0.0.0', port=5000)
