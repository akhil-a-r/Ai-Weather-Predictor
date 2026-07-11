from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from twilio.rest import Client
import os

app = Flask(__name__)

# ==========================================
# 1. TWILIO INFRASTRUCTURE CONFIGURATION
# ==========================================
# Replace these strings with your verified Twilio environment credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'your_account_sid_here')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'your_auth_token_here')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', 'your_twilio_phone_number_here')

# ==========================================
# 2. SUBSCRIBER TELEMETRY DATA STRUCTURE (MOCK DB)
# ==========================================
# Contains regional coordinates for target users subscribed to alert vectors.
SUBSCRIBED_USERS = [
    {"name": "Alice (Tropical Node 1)", "phone": "+12345678901", "subscribed_lat": 9.93},
    {"name": "Bob (Tropical Node 2)", "phone": "+19876543210", "subscribed_lat": 10.00},
    {"name": "Charlie (Temperate Node)", "phone": "+15555555555", "subscribed_lat": 40.71}
]

# ==========================================
# 3. AI PIPELINE MECHANICS (SCIKIT-LEARN)
# ==========================================
# Training Features Matrix: [Latitude, Longitude, Microclimate Humidity Bias]
X_train = np.array([
    [9.93, 76.26, 85],    # Equatorial Coastal Zone (High Precip Potential)
    [10.00, 76.30, 90],   # Dense Canopy Monsoonal Hub
    [40.71, -74.00, 40],  # Mid-Latitude Temperate Strip
    [34.05, -118.24, 15], # Subtropical Arid High Pressure Node
    [51.50, -0.12, 70]    # Maritime Polar System Matrix
])

# Target Core Array: Probability Matrix for Precipitation (0.0% - 100.0%)
y_train = np.array([88, 92, 35, 5, 65])

# Instantiating and executing fit optimizations on our Linear Predictor
ai_core = LinearRegression()
ai_core.fit(X_train, y_train)

# ==========================================
# 4. CRITICAL EMERGENCY MASS DISPATCH PIPELINE
# ==========================================
def dispatch_mass_emergency_sms(target_lat, hazard_condition):
    """
    Scans internal subscriber nodes to find populations located near the affected 
    latitude threshold and routes priority outbound emergency notification texts via Twilio.
    """
    alerts_fired_count = 0
    LATITUDE_PROXIMITY_RADIUS = 1.5  # Boundary check window in degrees
    
    for user in SUBSCRIBED_USERS:
        # Check if user falls inside the geographical tracking target window
        if abs(user["subscribed_lat"] - target_lat) <= LATITUDE_PROXIMITY_RADIUS:
            alert_message = (
                f"⚠️ SYSTEM HAZARD WARNING: {user['name']}\n"
                f"Core-AI has detected a severe weather vector at your coordinate path ({target_lat}°).\n"
                f"Status: {hazard_condition}.\n"
                f"Action Recommended: Monitor local channels and secure equipment."
            )
            
            try:
                # To activate live production SMS routing, ensure valid credentials 
                # are present and uncomment the block execution below:
                # -------------------------------------------------------------
                # client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                # client.messages.create(
                #     body=alert_message,
                #     from_=TWILIO_PHONE_NUMBER,
                #     to=user["phone"]
                # )
                # -------------------------------------------------------------
                print(f"📡 [EMERGENCY SMS DISPATCHED -> {user['phone']}]:\n{alert_message}\n")
                alerts_fired_count += 1
            except Exception as twilio_err:
                print(f"❌ Twilio Gateway Failure for user {user['name']}: {twilio_err}")
                
    return alerts_fired_count

# ==========================================
# 5. CORE WEB APPLICATION ROUTING ENDPOINTS
# ==========================================
@app.route('/')
def index():
    """Renders the primary mapping dashboard UI."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def run_predictive_inference():
    """
    Accepts spatial coordinate payloads from the interface layer, evaluates 
    climatological features through the AI model, and activates warnings if needed.
    """
    try:
        payload = request.get_json() or {}
        lat = float(payload.get('lat', 0.0))
        lng = float(payload.get('lng', 0.0))
        
        # Calculate environmental variables based on distance from the equator
        simulated_humidity_bias = 85 if abs(lat) < 20 else 50
        
        # Format payload tensors for model consumption
        inference_tensor = np.array([[lat, lng, simulated_humidity_bias]])
        raw_prediction = ai_core.predict(inference_tensor)[0]
        
        # Normalize and restrict prediction output to logical bounds (0-100%)
        rain_chance = max(0.0, min(100.0, round(float(raw_prediction), 1)))
        
        # Initialize default telemetry flags
        sms_broadcast_active = False
        dispatched_sms_count = 0
        
        # Operational condition state matrix
        if rain_chance > 75.0:
            condition_text = "🌧️ Severe Weather System Detected"
            advice_text = "CRITICAL: Heavy storm fronts actively forming. Automated broadcast protocol triggered."
            
            # Fire the SMS pipeline
            dispatched_sms_count = dispatch_mass_emergency_sms(lat, condition_text)
            if dispatched_sms_count > 0:
                sms_broadcast_active = True
        elif rain_chance > 45.0:
            condition_text = "☁️ Unstable / Overcast Microclimate"
            advice_text = "Atmospheric pressure dropping. Cloud density increasing. No emergency protocols needed."
        else:
            condition_text = "☀️ Atmosphere Nominal / Clear"
            advice_text = "High stability indices calculated. Clear conditions observed across this sector."

        return jsonify({
            'lat': lat,
            'lng': lng,
            'rain_chance': f"{rain_chance}%",
            'condition': condition_text,
            'advice': advice_text,
            'sms_triggered': sms_broadcast_active,
            'sms_count': dispatched_sms_count
        })

    except ValueError:
        return jsonify({'error': 'Malformed telemetry coordinate package received.'}), 400
    except Exception as general_err:
        return jsonify({'error': f'Pipeline runtime fault encountered: {str(general_err)}'}), 500

if __name__ == '__main__':
    # Start the Flask microservices runtime environment loop
    app.run(host='127.0.0.1', port=5000, debug=True)