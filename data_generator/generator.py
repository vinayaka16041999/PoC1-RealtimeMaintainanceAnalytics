import numpy as np
import pandas as pd
import json
import time
import redis
from datetime import datetime

# Connect to Redis
redis_client = redis.Redis(host='redis', port=6379, db=0)

def generate_sensor_reading():
    while True:
        timestamp = datetime.now().isoformat()
        vibration = np.random.normal(10, 2) * (1 + np.random.random() * 0.1)
        temperature = np.random.normal(35, 5) * (1 + np.random.random() * 0.05)
        pressure = np.random.normal(100, 10) * (1 + np.random.random() * 0.08)

        if np.random.random() < 0.05:  # 5% chance of anomaly
            vibration *= np.random.uniform(1.5, 3)
            temperature *= np.random.uniform(1.2, 2)
            pressure *= np.random.uniform(1.3, 2.5)

        reading = {
            'timestamp': timestamp,
            'vibration': float(vibration),
            'temperature': float(temperature),
            'pressure': float(pressure),
            'machine_id': f'MACH-{np.random.randint(1, 4)}'
        }

        # Publish to Redis
        redis_client.publish('sensor_data', json.dumps(reading))
        time.sleep(1)  # Simulate real-time data every second

if __name__ == "__main__":
    generate_sensor_reading()