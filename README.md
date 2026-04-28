# Mumbai Air - AQI Prediction

Daily air quality index (AQI) forecasting for Mumbai using OLS Linear Regression.
Website Link: https://mumbai-aqi-predictor.onrender.com/

## Features

- **Live AQI Data** — Real-time AQI from CPCB API
- **Weather Data** — Current conditions from OpenMeteo
- **Daily Forecast** — Predict tomorrow's AQI
- **Color-Coded Categories** — 6 AQI categories with health advisories

## Model

- **Algorithm:** OLS Linear Regression (statsmodels)
- **Features:** AQI lags (1, 2, 7 days), rolling 7-day average, weather (temp, humidity, wind, precipitation), season, day of week
- **Training Data:** 2+ years of historical Mumbai AQI

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone repo:
```bash
git clone <repo-url>
cd aqi-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare last 7 days AQI (from your notebook):
```python
import json
last_7_aqi = df_model['AQI'].tail(7).values.tolist()
with open('last_7_aqi.json', 'w') as f:
    json.dump(last_7_aqi, f)
```

4. Ensure these files are in project root:
   - `aqi_model.pkl`
   - `aqi_scaler.pkl`
   - `aqi_features.pkl`
   - `last_7_aqi.json`

### Local Development

```bash
waitress-serve --port=5000 app:app
```

Visit: `http://localhost:5000`

### Deployment (Render)

1. Push to GitHub
2. Connect repo to Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `waitress-serve --port=$PORT app:app`
5. Deploy

## API Endpoints

- `GET /` — Web interface
- `GET /weather` — Current weather (temp, humidity, wind, precipitation)
- `GET /aqi-today` — Today's AQI from CPCB
- `GET /history` — Last 7 days AQI
- `POST /predict` — Predict tomorrow's AQI (requires feature dict)

## File Structure

```
aqi-project/
├── app.py                  # Flask backend
├── data/.xlsx files        # Data Files
├── templates/index.html    # Frontend
├── requirements.txt        # Dependencies
├── Procfile               # Deployment config
├── .gitignore             # Git ignore rules
├── last_7_aqi.json        # Historical data
├── aqi_model.pkl          # Trained model
├── aqi_scaler.pkl         # StandardScaler
└── aqi_features.pkl       # Feature names
```

## Technologies

- **Backend:** Flask, statsmodels (OLS)
- **Frontend:** HTML, Tailwind CSS, Lucide Icons
- **APIs:** OpenMeteo (weather), CPCB (AQI)
- **Deployment:** Render
