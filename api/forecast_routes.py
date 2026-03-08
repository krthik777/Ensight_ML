from flask import Blueprint, request, jsonify
from services.month_forecast import month_forecast_service

forecast_bp = Blueprint('forecast', __name__)

@forecast_bp.route('/predict-month-end', methods=['POST'])
def predict_month_end():
    """Endpoint 4: Month-end unit and cost forecasting."""
    try:
        data = request.get_json()
        if not data or 'current_units' not in data or 'days_elapsed' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'current_units' or 'days_elapsed' in request body"
            }), 400
            
        current_units = data.get('current_units')
        days_elapsed = data.get('days_elapsed')
        
        result = month_forecast_service.predict_month_end(current_units, days_elapsed)
        
        if "error" in result:
             return jsonify({"success": False, "error": result["error"]}), 400
             
        result["success"] = True
        return jsonify(result), 200
        
    except Exception as e:
         return jsonify({
            "success": False,
            "error": str(e)
        }), 500
