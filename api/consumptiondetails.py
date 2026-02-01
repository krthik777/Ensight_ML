from flask import Blueprint, request, jsonify
import numpy as np
from datetime import datetime
from .nilm_integration import predict_appliances  # Import from same package

# Create Blueprint for consumption endpoints
consumption_bp = Blueprint('consumption', __name__)

# Consumption prediction functions
REALISTIC_MONTHLY_PATTERNS = {
    "weekday_pattern": {
        "00:00-06:00": {"base_load": 150, "active_appliances": ["fridge", "water_filter"]},
        "06:00-09:00": {"base_load": 400, "active_appliances": ["kitchen_outlets", "water_motor", "television"]},
        "09:00-17:00": {"base_load": 300, "active_appliances": ["fridge", "laptop_computer"]},
        "17:00-22:00": {"base_load": 800, "active_appliances": ["kitchen_outlets", "television", "air_conditioner"]},
        "22:00-00:00": {"base_load": 200, "active_appliances": ["fridge", "television"]}
    },
    "weekend_pattern": {
        "00:00-08:00": {"base_load": 180, "active_appliances": ["fridge", "water_filter"]},
        "08:00-12:00": {"base_load": 600, "active_appliances": ["kitchen_outlets", "water_motor", "television", "washing_machine"]},
        "12:00-18:00": {"base_load": 500, "active_appliances": ["air_conditioner", "television", "laptop_computer"]},
        "18:00-23:00": {"base_load": 900, "active_appliances": ["kitchen_outlets", "television", "air_conditioner"]},
        "23:00-00:00": {"base_load": 250, "active_appliances": ["fridge", "television"]}
    }
}

def calculate_realistic_baseline():
    """Calculate average power from realistic patterns"""
    total_power = 0
    total_hours = 0
    
    for day_type in ["weekday", "weekend"]:
        pattern = REALISTIC_MONTHLY_PATTERNS[f"{day_type}_pattern"]
        days_count = 20 if day_type == "weekday" else 10
        
        for time_slot, data in pattern.items():
            hours = 6 if "06:00" in time_slot else 4  # Simplified hour calculation
            total_power += data["base_load"] * hours * days_count
            total_hours += hours * days_count
    
    return (total_power / total_hours) / 1000

def calculate_kseb_cost(total_units):
    """KSEB billing calculation"""
    cost = 0
    units_remaining = total_units
    
    if units_remaining > 500:
        cost += (units_remaining - 500) * 7.10
        units_remaining = 500
    if units_remaining > 400:
        cost += (units_remaining - 400) * 7.00
        units_remaining = 400
    if units_remaining > 300:
        cost += (units_remaining - 300) * 6.70
        units_remaining = 300
    if units_remaining > 200:
        cost += (units_remaining - 200) * 5.20
        units_remaining = 200
    if units_remaining > 100:
        cost += (units_remaining - 100) * 4.00
        units_remaining = 100
    cost += units_remaining * 3.30
    
    cost += 20  # Fixed charge
    cost *= 1.15  # Electricity duty
    return cost

def calculate_appliance_level_costs(current_appliances, monthly_kwh, total_cost):
    """Calculate cost breakdown per appliance"""
    appliance_costs = {}
    total_current_power = sum(appliance["power"] for appliance in current_appliances.values() if appliance["state"] == "ON")
    
    for appliance_name, appliance_data in current_appliances.items():
        if appliance_data["state"] == "ON":
            power_ratio = appliance_data["power"] / total_current_power if total_current_power > 0 else 0
            appliance_consumption = monthly_kwh * power_ratio
            appliance_cost = calculate_kseb_cost(appliance_consumption)
            
            appliance_costs[appliance_name] = {
                "estimatedMonthlyConsumption": round(appliance_consumption, 2),
                "estimatedMonthlyCost": round(appliance_cost, 2),
                "currentPower": appliance_data["power"],
                "contributionPercentage": round(power_ratio * 100, 1)
            }
    
    return appliance_costs

def calculate_smart_monthly_estimate(current_appliances, days_elapsed=0):
    """Calculate monthly consumption with realistic patterns"""
    current_power = sum(appliance["power"] for appliance in current_appliances.values() if appliance["state"] == "ON")
    current_kw = current_power / 1000
    
    realistic_baseline = calculate_realistic_baseline()
    
    if days_elapsed > 0:
        actual_weight = min(days_elapsed / 30, 0.7)
        realistic_weight = 1 - actual_weight
        blended_usage_rate = (current_kw * actual_weight) + (realistic_baseline * realistic_weight)
    else:
        blended_usage_rate = realistic_baseline
    
    monthly_kwh = blended_usage_rate * 720  # 30 days * 24 hours
    total_cost = calculate_kseb_cost(monthly_kwh)
    appliance_costs = calculate_appliance_level_costs(current_appliances, monthly_kwh, total_cost)
    
    return {
        "estimatedMonthlyConsumption": round(monthly_kwh, 2),
        "estimatedMonthlyCost": round(total_cost, 2),
        "applianceBreakdown": appliance_costs,
        "predictionMethod": "blended" if days_elapsed > 0 else "realistic_baseline",
        "confidence": min(0.3 + (days_elapsed / 30 * 0.7), 0.95)
    }

def predict_appliances_from_sequence(mains_sequence, room_id=None):
    """
    Call your actual NILM model from the moved file
    """
    return predict_appliances(mains_sequence, room_id)  # Your real function

# Consumption prediction endpoint
@consumption_bp.route('/nilm/consumption-prediction', methods=['POST'])
def consumption_prediction():
    """
    Endpoint for monthly consumption and cost predictions
    Expects JSON with mains power sequence
    """
    try:
        data = request.get_json()
        
        if not data:
             return jsonify({
                "success": False,
                "error": "Missing request body"
            }), 400

        mains_sequence = data.get('mainsSequence')
        room_id = data.get('roomId', 'unknown')
        user_id = data.get('userId', 'unknown')
        days_elapsed = data.get('daysElapsed', 0)  # Days of data collected so far

        if not mains_sequence and room_id == 'unknown':
            return jsonify({
                "success": False,
                "error": "Must provide either mainsSequence or roomId"
            }), 400
        
        # Get appliance predictions first (using your existing model)
        appliance_predictions = predict_appliances_from_sequence(mains_sequence, room_id)
        
        # Calculate monthly consumption and cost predictions
        monthly_estimation = calculate_smart_monthly_estimate(
            appliance_predictions["appliances"], 
            days_elapsed
        )
        
        response = {
            "success": True,
            "data": {
                "timestamp": datetime.now().isoformat(),
                "roomId": room_id,
                "userId": user_id,
                "currentUsage": {
                    "totalPower": appliance_predictions["totalPower"],
                    "activeAppliances": appliance_predictions["activeAppliances"],
                    "applianceCount": len(appliance_predictions["activeAppliances"])
                },
                "monthlyPrediction": monthly_estimation,
                "billingCycle": {
                    "daysElapsed": days_elapsed,
                    "daysRemaining": 30 - days_elapsed,
                    "predictionConfidence": monthly_estimation["confidence"]
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500

# Additional endpoint for consumption history (if needed)
@consumption_bp.route('/nilm/consumption-history', methods=['GET'])
def consumption_history():
    """Get consumption history for a user"""
    user_id = request.args.get('userId')
    days = request.args.get('days', 30, type=int)
    
    # TODO: Implement actual history retrieval
    return jsonify({
        "success": True,
        "data": {
            "userId": user_id,
            "periodDays": days,
            "consumptionHistory": []  # Placeholder
        }
    })