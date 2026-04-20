from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import sys
import threading
import time
from datetime import date
import pandas as pd
import numpy as np
import requests
import json

app = Flask(__name__)
CORS(app)

# Patch patsy to handle missing frame context when unpickling statsmodels OLS models
def _patch_patsy():
    try:
        import patsy.eval as pe
        _orig = pe.EvalEnvironment.capture.__func__

        @classmethod
        def _safe_capture(cls, eval_env=0, reference=0):
            try:
                frame = sys._getframe(reference + 1)
                if frame is None:
                    raise AttributeError
                return _orig(cls, eval_env, reference)
            except (AttributeError, ValueError):
                return cls([{}, {}])

        pe.EvalEnvironment.capture = _safe_capture
    except Exception:
        pass

_patch_patsy()

# Load model, scaler, features
try:
    with open('aqi_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('aqi_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('aqi_features.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
except Exception:
    model = scaler = feature_cols = None

# Load last 7 days AQI history
HISTORY_FILE = 'last_7_aqi.json'

def load_history():
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {'values': data, 'last_updated': None}
        return data
    except Exception:
        return {'values': [85, 85, 85, 85, 85, 85, 85], 'last_updated': None}

history_data = load_history()
last_7_aqi = history_data['values']

def fetch_current_aqi():
    """Fetch today's AQI from data.gov.in API."""
    try:
        API_KEY = '579b464db66ec23bdd000001af2dae7288454bc7700cb29a81207cb2'
        url = 'https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69'
        params = {'api-key': API_KEY, 'format': 'json', 'limit': 500, 'filters[city]': 'Mumbai'}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if 'records' not in data or len(data['records']) == 0:
            print(f'[AQI API] No records returned. Response keys: {list(data.keys())}')
            return None

        api_data = pd.DataFrame(data['records'])
        api_data['avg_value'] = pd.to_numeric(api_data['avg_value'], errors='coerce')
        aqi = api_data.groupby('pollutant_id')['avg_value'].mean().max()

        if pd.isna(aqi) or aqi <= 0:
            print(f'[AQI API] Invalid AQI value computed: {aqi}')
            return None

        print(f'[AQI API] Successfully fetched AQI: {aqi}')
        return round(float(aqi), 1)
    except Exception as e:
        print(f'[AQI API] Error fetching current AQI: {e}')
        return None

def update_aqi_history():
    """Fetch today's AQI and roll the 7-day history forward. Saves to disk."""
    global last_7_aqi

    today = str(date.today())
    current = load_history()

    # Skip if already updated today
    if current.get('last_updated') == today:
        return

    aqi_value = fetch_current_aqi()
    if aqi_value is None:
        return

    updated = current['values'][1:] + [aqi_value]

    new_data = {'values': updated, 'last_updated': today}
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(new_data, f)
        last_7_aqi = updated
        print(f'[AQI History] Updated for {today}: {updated}')
    except Exception as e:
        print(f'[AQI History] Failed to save: {e}')

def daily_updater():
    """Background thread: update AQI history once per day."""
    # Run once on startup
    update_aqi_history()
    while True:
        # Sleep 24 hours then update again
        time.sleep(86400)
        update_aqi_history()

# Start the background updater thread (daemon so it dies when the app stops)
updater_thread = threading.Thread(target=daily_updater, daemon=True)
updater_thread.start()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weather', methods=['GET'])
def get_weather():
    try:
        res = requests.get(
            'https://api.open-meteo.com/v1/forecast',
            params={
                'latitude': 19.076,
                'longitude': 72.8777,
                'current': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation',
                'timezone': 'Asia/Kolkata'
            },
            timeout=10
        )
        data = res.json()['current']
        return jsonify({
            'temp': data['temperature_2m'],
            'humidity': data['relative_humidity_2m'],
            'windspeed': data['wind_speed_10m'],
            'precipitation': data['precipitation'],
            'source': 'live'
        })
    except Exception as e:
        print(f'[Weather API] Error fetching weather: {e}')
        return jsonify({
            'temp': None,
            'humidity': None,
            'windspeed': None,
            'precipitation': None,
            'source': 'unavailable'
        })

@app.route('/aqi-today', methods=['GET'])
def get_today_aqi():
    try:
        aqi = fetch_current_aqi()
        if aqi is not None:
            return jsonify({'aqi': aqi, 'source': 'live'})

        # Fallback: use the most recent real AQI from history
        current = load_history()
        history_values = current.get('values', [])
        if history_values:
            fallback_aqi = history_values[-1]
            print(f'[AQI Today] API unavailable, using last known AQI: {fallback_aqi}')
            return jsonify({'aqi': fallback_aqi, 'source': 'history'})

        return jsonify({'aqi': None, 'source': 'unavailable'})
    except Exception as e:
        print(f'[AQI Today] Error: {e}')
        current = load_history()
        history_values = current.get('values', [])
        if history_values:
            return jsonify({'aqi': history_values[-1], 'source': 'history'})
        return jsonify({'aqi': None, 'source': 'unavailable'})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'history': last_7_aqi})

AQI_LAG_COLS = ['AQI_lag1', 'AQI_lag2', 'AQI_lag7', 'AQI_roll7']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'aqi': None, 'status': 'error', 'message': 'Model not loaded'}), 500

        data = request.json
        df = pd.DataFrame([data])[feature_cols]

        # Apply log1p to AQI lag features exactly as done during training
        for col in AQI_LAG_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(scaled, columns=feature_cols)

        # Model predicts log1p(AQI_next), so reverse with expm1
        log_pred = model.predict(df_scaled)[0]
        pred = np.expm1(log_pred)
        pred = max(0, min(500, pred))

        return jsonify({'aqi': round(pred, 1), 'status': 'success'})
    except Exception as e:
        return jsonify({'aqi': None, 'status': 'error', 'message': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    current = load_history()
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'history_last_updated': current.get('last_updated'),
        'history': last_7_aqi
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
