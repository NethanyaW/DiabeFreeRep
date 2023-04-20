from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, auth
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Set up Firebase credentials and initialize app
cred = credentials.Certificate('C:/Users/HP/Desktop/DiabeFreeAI/serviceAccountKey.json')
firebase_admin.initialize_app(cred)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_blood_glucose():
    # Get the data from the request
    data = request.get_json()

    # Save the data to Firestore
    db = firestore.client()
    user = auth.current_user()
    user_ref = db.collection('users').document(user.uid).collection('BG')
    user_ref.add(data)

    # Check if we have enough data to train the model
    num_records = user_ref.get().size
    if num_records >= 10:
        # Retrieve the data from Firestore
        bg_data = []
        timestamps = []
        for doc in user_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).stream():
            bg_data.append(doc.to_dict()['level'])
            timestamps.append(doc.to_dict()['timestamp'])

        # Train the model
        model = train_model(bg_data, timestamps)

        # Predict the blood glucose level for the next entry
        predicted_level = predict(model, bg_data[-1], timestamps[-1])

        # Return the predicted blood glucose level to the user
        return {'predicted_level': float(predicted_level)}

    else:
        return {'status': 'success', 'message': 'Data saved successfully. We need at least 10 entries to train the model.'}

def train_model(glucose_levels, times):
    # Convert the timestamps to numerical values
    times = np.array([(t - times[-1]).total_seconds() / 3600.0 for t in times]).reshape(-1, 1)

    # Train a linear regression model using the blood glucose levels and times
    model = LinearRegression()
    model.fit(times, glucose_levels)

    return model

def predict(model, bg_data, timestamps):
    # Convert the timestamp to a numerical value
    timestamp_num = (datetime.strptime(timestamps[-1], '%d %B %Y at %H:%M:%S %z') - datetime(1970, 1, 1)).total_seconds() / 3600.0

    # Use the trained model to predict the blood glucose level at the given timestamp
    predicted_level = model.predict(np.array([timestamp_num]).reshape(-1, 1))[0]

    return predicted_level