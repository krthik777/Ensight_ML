class CostPredictionService:
    @staticmethod
    def calculate_monthly_cost(daily_usage_kwh):
        """Endpoint 3: Predict cost based on daily usage"""
        try:
            daily_usage_kwh = float(daily_usage_kwh)
            
            # Estimate monthly units
            estimated_monthly_units = daily_usage_kwh * 30
            
            # KSEB Slab Calculation
            cost = 0
            units_remaining = estimated_monthly_units
            
            if units_remaining > 500:
                cost += (units_remaining - 500) * 7.10
                units_remaining = 500
            if units_remaining > 300:
                cost += (units_remaining - 300) * 6.50
                units_remaining = 300
            if units_remaining > 200:
                cost += (units_remaining - 200) * 5.20
                units_remaining = 200
            if units_remaining > 100:
                cost += (units_remaining - 100) * 4.00
                units_remaining = 100
            if units_remaining > 0:
                cost += units_remaining * 3.30
                
            cost += 20  # Fixed charge
            cost *= 1.15  # Electricity duty
            
            return {
                "estimated_monthly_units": round(estimated_monthly_units, 2),
                "estimated_cost": round(cost, 2),
                "currency": "INR"
            }
        except (ValueError, TypeError):
            return {"error": "Invalid input for daily_usage_kwh"}

cost_prediction_service = CostPredictionService()
