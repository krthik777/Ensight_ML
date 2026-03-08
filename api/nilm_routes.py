from flask import Blueprint, request, jsonify
from services.nilm_inference import nilm_inference_service

nilm_bp = Blueprint('nilm', __name__)

@nilm_bp.route('/detect-appliances', methods=['POST'])
def detect_appliances():
    """Endpoint 1: Detect which appliances are ON"""
    try:
        data = request.get_json()
        if not data or 'mains_sequence' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'mains_sequence' in request body"
            }), 400
            
        mains_sequence = data.get('mains_sequence')
        
        result = nilm_inference_service.detect_appliances(mains_sequence)
        
        # Merge success flag if missing, or handle specific fallback format
        if "success" not in result:
             result["success"] = True
             
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@nilm_bp.route('/predict-appliance-power', methods=['POST'])
def predict_appliance_power():
    """Endpoint 2: Predict how much power each appliance uses"""
    try:
        data = request.get_json()
        if not data or 'mains_sequence' not in data:
             return jsonify({
                "success": False,
                "error": "Missing 'mains_sequence' in request body"
            }), 400
            
        mains_sequence = data.get('mains_sequence')
        
        result = nilm_inference_service.predict_appliance_power(mains_sequence)
        
        if "success" not in result:
             result["success"] = True
             
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@nilm_bp.route('/detect-anomaly', methods=['POST'])
def detect_anomaly():
    """Endpoint 8: Detect unusual power spikes"""
    try:
        data = request.get_json()
        if not data or 'mains_sequence' not in data:
             return jsonify({
                "success": False,
                "error": "Missing 'mains_sequence' in request body"
            }), 400
            
        mains_sequence = data.get('mains_sequence')
        
        result = nilm_inference_service.detect_anomaly(mains_sequence)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
