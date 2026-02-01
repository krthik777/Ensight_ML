from flask import Flask, request, jsonify
from api.consumptiondetails import consumption_bp
from api.nilm_integration import predict_appliances  # Updated import

app = Flask(__name__)

# Register the consumption blueprint
app.register_blueprint(consumption_bp)

# Your existing routes
@app.route('/')
def hello():
    return "NILM API Server is running!"

# Your existing NILM prediction endpoint
@app.route('/nilm/predict', methods=['POST'])
def nilm_predict():
    """
    Main NILM prediction endpoint
    """
    try:
        data = request.get_json()
        
        mains_sequence = data.get('mainsSequence')
        room_id = data.get('roomId')
        user_id = data.get('userId', 'unknown')

        if not mains_sequence and not room_id:
            return jsonify({
                "success": False,
                "error": "Must provide either mainsSequence or roomId"
            }), 400
        
        # Call the actual NILM model from the moved file
        prediction_result = predict_appliances(mains_sequence, room_id)
        
        response = {
            "success": True,
            "data": {
                "backendStatus": "success",
                "prediction": prediction_result,
                "summary": {
                    "activeCount": len(prediction_result.get("activeAppliances", [])),
                    "totalConsumption": prediction_result.get("totalPower", 0),
                    "estimatedCost": round(prediction_result.get("totalPower", 0) * 0.0065, 2)  # Simple cost estimate
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "NILM API",
        "timestamp": "2025-10-21T14:30:45.123456"
    })

if __name__ == '__main__':
    print("🚀 Starting Single Server NILM API")
    print("🔌 Endpoints:")
    print("   POST /nilm/predict - Main appliance prediction")
    print("   POST /nilm/consumption-prediction - Monthly consumption estimates")
    print("   GET  /health - Service health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)