import numpy as np
import pandas as pd
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import deque
import logging
import os
import random

# Connect to Redis
redis_client = redis.Redis(host='redis', port=6379, db=0)

# Initialize models and data storage
scaler = StandardScaler()
model = IsolationForest(contamination=0.01, random_state=42)  # Adjusted contamination as per previous suggestions
historical_data = deque(maxlen=10000)  # Store last 10,000 readings for retraining
last_retrain_time = datetime.now()
RETRAIN_INTERVAL_DAYS = 3
INITIAL_FIT_SIZE = 100  # Number of initial readings to fit the models

logging.basicConfig(filename='/app/alerts.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def fit_initial_models():
    global scaler, model
    print("Fitting initial models with first 100 readings...")
    
    # Wait for initial data to accumulate
    initial_data = []
    pubsub = redis_client.pubsub()
    pubsub.subscribe('sensor_data')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            reading = json.loads(message['data'].decode('utf-8'))
            initial_data.append(reading)
            if len(initial_data) >= INITIAL_FIT_SIZE:
                break
    
    if not initial_data:
        raise ValueError("No initial data received to fit models")

    # Convert to DataFrame for fitting
    initial_df = pd.DataFrame(initial_data)
    initial_df = initial_df.sort_values('timestamp')

    # Feature engineering for initial data
    initial_df['vib_rolling_mean'] = initial_df['vibration'].rolling(window=48, min_periods=1).mean()  # Larger window
    initial_df['temp_rolling_mean'] = initial_df['temperature'].rolling(window=48, min_periods=1).mean()
    initial_df['press_rolling_mean'] = initial_df['pressure'].rolling(window=48, min_periods=1).mean()

    features = ['vibration', 'temperature', 'pressure', 'vib_rolling_mean', 'temp_rolling_mean', 'press_rolling_mean']
    X_initial = initial_df[features].fillna(0)

    # Fit scaler and model
    scaler.fit(X_initial)
    model.fit(scaler.transform(X_initial))
    print("Initial models fitted successfully")

def process_reading(reading):
    global scaler, model, historical_data, last_retrain_time

    # Convert reading to DataFrame for processing
    df = pd.DataFrame([reading])

    # Feature engineering (simple for streaming)
    df['vib_rolling_mean'] = df['vibration'].rolling(window=48, min_periods=1).mean()  # Larger window
    df['temp_rolling_mean'] = df['temperature'].rolling(window=48, min_periods=1).mean()
    df['press_rolling_mean'] = df['pressure'].rolling(window=48, min_periods=1).mean()

    # Prepare features for prediction
    features = ['vibration', 'temperature', 'pressure', 'vib_rolling_mean', 'temp_rolling_mean', 'press_rolling_mean']
    X = df[features].fillna(0)

    # Transform using the fitted scaler
    X_transformed = scaler.transform(X)

    # Predict anomalies
    anomaly_score = model.decision_function(X_transformed)[0]
    is_anomaly = 1 if anomaly_score < 0 else 0

    # Calculate health score
    health_score = 100 - (
        abs(df['vibration'].iloc[0] - df['vib_rolling_mean'].iloc[0]) * 0.3 +
        abs(df['temperature'].iloc[0] - df['temp_rolling_mean'].iloc[0]) * 0.3 +
        abs(df['pressure'].iloc[0] - df['press_rolling_mean'].iloc[0]) * 0.2 +
        abs(anomaly_score) * 0.2
    ) * 10

    weight = random.randint(1, 70)*0.01
    if is_anomaly:
        health_score = health_score * weight
    else:
        health_score = health_score

    # Check for alerts and publish to Redis
    if is_anomaly and health_score < 70:  # Adjusted alert logic as per previous suggestions
        alert_message = f"ALERT at {reading['timestamp']}: Machine {reading['machine_id']} - Health Score: {health_score:.2f}, Anomaly: {bool(is_anomaly)}"
        redis_client.publish('alerts_channel', alert_message)  # Publish to Redis
        logging.info(alert_message)  # Log to file
        print(alert_message)  # Still print to console for debugging

    # Store historical data
    historical_data.append(reading)

    # Retrain model if enough time has passed
    if (datetime.now() - last_retrain_time).days >= RETRAIN_INTERVAL_DAYS:
        retrain_model()
        last_retrain_time = datetime.now()

    return {'health_score': health_score, 'anomaly': is_anomaly, 'anomaly_score': anomaly_score}

def retrain_model():
    global scaler, model, historical_data

    if len(historical_data) < 100:  # Minimum data for retraining
        return

    # Convert historical data to DataFrame
    hist_df = pd.DataFrame(historical_data)
    hist_df = hist_df.sort_values('timestamp')

    # Feature engineering
    hist_df['vib_rolling_mean'] = hist_df['vibration'].rolling(window=48, min_periods=1).mean()  # Larger window
    hist_df['temp_rolling_mean'] = hist_df['temperature'].rolling(window=48, min_periods=1).mean()
    hist_df['press_rolling_mean'] = hist_df['pressure'].rolling(window=48, min_periods=1).mean()

    features = ['vibration', 'temperature', 'pressure', 'vib_rolling_mean', 'temp_rolling_mean', 'press_rolling_mean']
    X = hist_df[features].fillna(0)

    # Retrain scaler and model
    scaler.fit(X)
    model.fit(scaler.transform(X))
    print(f"Model retrained at {datetime.now().isoformat()} with {len(historical_data)} samples")


if __name__ == "__main__":
    # Fit initial models before starting to process
    fit_initial_models()

    pubsub = redis_client.pubsub()
    pubsub.subscribe('sensor_data')

    data_window = deque(maxlen=300)  # Window for visualization

    for message in pubsub.listen():
        if message['type'] == 'message':
            reading = json.loads(message['data'].decode('utf-8'))
            result = process_reading(reading)
            reading.update(result)
            data_window.append(reading)