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

def calculate_energy_charge(units):
    """Calculate energy charge based on KSEB 2025-2027 tariff"""
    cost = 0
    if units <= 250:
        # Telescopic
        if units > 200:
            cost += (units - 200) * 8.50
            units = 200
        if units > 150:
            cost += (units - 150) * 7.20
            units = 150
        if units > 100:
            cost += (units - 100) * 5.35
            units = 100
        if units > 50:
            cost += (units - 50) * 4.25
            units = 50
        cost += units * 3.35
    else:
        # Non-telescopic
        if units <= 300:
            cost = units * 6.75
        elif units <= 350:
            cost = units * 7.60
        elif units <= 400:
            cost = units * 7.95
        elif units <= 500:
            cost = units * 8.25
        else:
            cost = units * 9.20
    return cost

def calculate_fixed_charge(units):
    """Calculate fixed charge based on KSEB 2025-2027 tariff"""
    if units <= 50: return 50
    if units <= 100: return 85
    if units <= 150: return 105
    if units <= 200: return 140
    if units <= 250: return 160
    if units <= 300: return 220
    if units <= 350: return 240
    if units <= 400: return 260
    if units <= 500: return 285
    return 310

def calculate_total_bill(units):
    """Generate structured response for KSEB bill"""
    energy_charge = calculate_energy_charge(units)
    fixed_charge = calculate_fixed_charge(units)
    electricity_duty = energy_charge * 0.10
    total_estimated_cost = energy_charge + fixed_charge + electricity_duty
    
    return {
        "monthly_units": round(units, 2),
        "energy_charge": round(energy_charge, 2),
        "fixed_charge": round(fixed_charge, 2),
        "electricity_duty": round(electricity_duty, 2),
        "total_estimated_cost": round(total_estimated_cost, 2)
    }

def calculate_appliance_level_costs(current_appliances, monthly_kwh, total_cost):
    """Calculate cost breakdown per appliance"""
    appliance_costs = {}
    total_current_power = sum(appliance["power"] for appliance in current_appliances.values() if appliance["state"] == "ON")
    
    for appliance_name, appliance_data in current_appliances.items():
        if appliance_data["state"] == "ON":
            power_ratio = appliance_data["power"] / total_current_power if total_current_power > 0 else 0
            appliance_consumption = monthly_kwh * power_ratio
            appliance_bill = calculate_total_bill(appliance_consumption)
            
            appliance_costs[appliance_name] = {
                "estimatedMonthlyConsumption": round(appliance_consumption, 2),
                "estimatedMonthlyCost": round(appliance_bill["total_estimated_cost"], 2),
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
    bill_details = calculate_total_bill(monthly_kwh)
    total_cost = bill_details["total_estimated_cost"]
    appliance_costs = calculate_appliance_level_costs(current_appliances, monthly_kwh, total_cost)
    
    return {
        "estimatedMonthlyConsumption": round(monthly_kwh, 2),
        "estimatedMonthlyCost": round(total_cost, 2),
        "billDetails": bill_details,
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