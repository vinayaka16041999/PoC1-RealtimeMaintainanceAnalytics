import redis
import threading  # Add this import
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Connect to Redis
redis_client = redis.Redis(host='redis', port=6379, db=0)

# Global list to store alerts
alerts = []

def update_alerts():
    global alerts
    pubsub = redis_client.pubsub()
    pubsub.subscribe('alerts_channel')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            alert = message['data'].decode('utf-8')
            alerts.append(alert)
            if len(alerts) > 100:
                alerts.pop(0)

# Start a thread to listen for alerts
threading.Thread(target=update_alerts, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html', alerts=alerts)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)