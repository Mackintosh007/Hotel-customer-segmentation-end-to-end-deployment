from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained KMeans model
try:
    model = pickle.load(open('kmeans_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'kmeans_model.pkl' not found. Please make sure the model file is in the same directory as app.py.")
    model = None

@app.route('/')
def index():
    # Renders the main UI from the 'templates' folder
    return render_template('ui.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle the case where the model failed to load
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded. Check server logs.'})

    data = request.get_json()
    try:
        # Prepare input features in the same order used during training
        # The keys here must match the data sent from the JavaScript on the front end
        input_features = np.array([[
            data['adults'],
            data['children'],
            data['lead_time'],
            data['adr'],
            data['total_nights'],
            data['total_of_special_requests']
        ]])
        
        # Make the prediction
        assigned_cluster = model.predict(input_features)[0]
        
        # Return the result as a JSON response
        return jsonify({'status': 'success', 'assigned_cluster': int(assigned_cluster)})
    except KeyError as e:
        return jsonify({'status': 'error', 'message': f'Missing data in request: {e}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Running in debug mode allows for automatic reloads on code changes
    # Ensure that your `templates` folder and `kmeans_model.pkl` are in the same directory as this file.
    app.run(debug=True)
