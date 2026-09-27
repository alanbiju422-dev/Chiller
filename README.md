# VOLTURA
AI Energy-Waste Auditor for Commercial Buildings

## Problem
Commercial buildings often suffer from hidden energy waste due to mechanical degradation, faulty settings, or suboptimal operations in heavy machinery like chillers and HVAC systems. Traditional systems only trigger threshold-based alarms which produce high noise, lacking context about the root cause and actual energy impact.

## Solution
Monitor
↓
Detect
↓
Explain
↓
Alert
↓
Investigate
↓
Quantify Savings
↓
ROI
↓
Report

## Key Features
- Intelligent energy monitoring
- Expected energy prediction
- Energy anomaly detection
- Contextual anomaly detection
- Investigation workspace
- AI-assisted investigation
- Real-time/replay monitoring
- Telegram alerts
- SMS alerts if actually implemented
- Energy savings estimation
- ROI calculation
- PDF/report generation

## Machine Learning
- **Random Forest Regressor**: Used for building baseline expectations based on non-linear feature interactions.
- **HistGradientBoosting Regressor**: Optimized gradient boosting used as the primary predictive model for continuous target variables (expected energy).
- **Isolation Forest**: Used for contextual, multi-dimensional anomaly detection identifying atypical operational behaviors.

## Dataset
- timestamp
- equipment_id (e.g., CHILLER-01)
- Chilled Water Rate (L/sec)
- Cooling Water Temperature (C)
- Building Load (RT)
- Chiller Energy Consumption (kWh)
- Outside Temperature (F)
- Dew Point (F)
- Humidity (%)
- Wind Speed (mph)
- Pressure (in)

## Architecture
- **data/**: Validation, chronological sorting, and train/unseen split.
- **ml/**: Pipeline with median imputation, feature engineering, predictive models, and Isolation Forest.
- **replay/**: Live simulated data engine operating cleanly on Streamlit's state.
- **alerts/**: Telegram and SMS integration for severe anomalies.
- **investigation/**: Generates deterministic recommendations and gathers context.
- **ai/**: Gemini AI for deep, automated analysis and pdf reporting.
- **tests/**: Unit tests protecting critical pipeline integrity.

## Project Structure
- pp.py: Streamlit entrypoint dashboard
- 	rain_pipeline.py: Pipeline for initial model training
- orecast_pipeline.py: Script/module for prediction logic
- i/: Contains AI integrations (Gemini analysis, PDF report generators)
- lerts/: Telegram and SMS alert workers
- data/: Dataset storage and validation logic
- ml/: Core machine learning modeling and data preparation modules

## Installation
`ash
git clone https://github.com/alanbiju422-dev/Chiller.git
cd Chiller

python -m venv venv

# Windows:
venv\Scripts\activate

pip install -r requirements.txt
`

## Environment Configuration
1. Copy .env.example to .env
2. Add the user's own API credentials
3. Never commit .env

Example:
`env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-1.5-flash
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
`

## Running the Application
`ash
python -m streamlit run app.py
`

## Telegram Alerts
The Telegram Bot API is used for alerts.
Configure via .env:
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID

## AI Analysis
Gemini Flash is used for AI-assisted investigation.
Configure via .env:
GEMINI_API_KEY
GEMINI_MODEL

## Energy Savings
Savings estimates are calculated based on the difference between expected and actual energy consumption over the duration of the anomaly multiplied by standardized energy costs.

## ROI
ROI estimates incorporate calculated cost savings extrapolated over operational lifespans versus typical maintenance costs.

## Scalability
FUTURE SCOPE: Expansion from chiller/HVAC monitoring to:
- AHUs
- pumps
- cooling towers
- compressors
- lighting
- other building equipment
- multiple buildings
- IoT/BMS/SCADA integrations

## Limitations
- Anomaly detection does not automatically prove a mechanical failure.
- Dataset limitations may restrict generalized accuracy.
- Predictions highly depend on available input data quality.

## Future Scope
- Integration with live BMS API streams
- Enhanced multi-modal alert routing
- Prescriptive maintenance dispatching

## License
License can be added later if appropriate.
