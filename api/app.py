from flask import Flask, jsonify
from api.nilm_routes import nilm_bp
from api.cost_routes import cost_bp
from api.forecast_routes import forecast_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(nilm_bp)      # Provides /detect-appliances, /predict-appliance-power, /detect-anomaly
app.register_blueprint(cost_bp)      # Provides /predict-cost
app.register_blueprint(forecast_bp)  # Provides /predict-month-end

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "EnSight Model Engine (AI)",
        "endpoints": [
            "/detect-appliances",
            "/predict-appliance-power",
            "/predict-cost",
            "/predict-month-end",
            "/detect-anomaly"
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "EnSight Data Inference API"
    })

if __name__ == '__main__':
    print("🚀 Starting EnSight AI Model Engine...")
    print("🔌 Endpoints available:")
    print("   POST /detect-appliances        - Returns active appliances & confidence")
    print("   POST /predict-appliance-power  - Returns discrete power levels (W) per appliance")
    print("   POST /predict-cost             - Returns monthly cost prediction based on daily avg")
    print("   POST /predict-month-end        - Returns end-of-month unit and cost forecasts")
    print("   POST /detect-anomaly           - Detects strange power spikes")
    
    # We run on 8000 to distinct from normal React/Node 3000/5000 conventions
    app.run(debug=True, host='0.0.0.0', port=8000)
