from flask import Blueprint, request, jsonify
from services.cost_prediction import cost_prediction_service

cost_bp = Blueprint('cost', __name__)

@cost_bp.route('/predict-cost', methods=['POST'])
def predict_cost():
    """Endpoint 3: Cost Prediction Endpoint using daily average."""
    try:
        data = request.get_json()
        if not data or 'daily_usage_kwh' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'daily_usage_kwh' in request body"
            }), 400
            
        daily_usage = data.get('daily_usage_kwh')
        
        result = cost_prediction_service.calculate_monthly_cost(daily_usage)
        
        if "error" in result:
             return jsonify({"success": False, "error": result["error"]}), 400
             
        result["success"] = True
        return jsonify(result), 200
        
    except Exception as e:
         return jsonify({
            "success": False,
            "error": str(e)
        }), 500
