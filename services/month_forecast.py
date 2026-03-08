class MonthForecastService:
    @staticmethod
    def predict_month_end(current_units, days_elapsed):
        """Endpoint 4: Month-end prediction"""
        try:
            current_units = float(current_units)
            days_elapsed = int(days_elapsed)
            
            if days_elapsed <= 0:
                return {"error": "days_elapsed must be greater than 0"}
                
            avg_per_day = current_units / days_elapsed
            predicted_month_units = avg_per_day * 30
            
            # Use CostPredictionService to calculate the cost
            from services.cost_prediction import cost_prediction_service
            
            # We pass the predicted average daily usage back to the cost calculator
            cost_prediction = cost_prediction_service.calculate_monthly_cost(avg_per_day)
            
            if "error" in cost_prediction:
                return cost_prediction
                
            return {
                "predicted_month_units": round(predicted_month_units, 2),
                "expected_cost": cost_prediction["estimated_cost"]
            }
        except (ValueError, TypeError):
             return {"error": "Invalid input for current_units or days_elapsed"}

month_forecast_service = MonthForecastService()
