"""
services/month_forecast.py
---------------------------
Month-end electricity consumption and cost forecasting.
"""


class MonthForecastService:

    @staticmethod
    def predict_month_end(current_units, days_elapsed, rate_per_kwh: float = 6.5):
        """
        Forecast end-of-month consumption and cost via linear extrapolation.

        Args:
            current_units : kWh consumed so far this month.
            days_elapsed  : Number of days elapsed in the current month.
            rate_per_kwh  : User's flat rate from MongoDB settings (INR/kWh).
                           Passed through to the cost calculator.
        """
        try:
            current_units = float(current_units)
            days_elapsed = int(days_elapsed)

            if days_elapsed <= 0:
                return {"error": "days_elapsed must be greater than 0"}

            avg_per_day = current_units / days_elapsed
            predicted_month_units = round(avg_per_day * 30, 2)

            from services.cost_prediction import cost_prediction_service

            cost_result = cost_prediction_service.calculate_monthly_cost(
                avg_per_day, rate_per_kwh=rate_per_kwh
            )

            if "error" in cost_result:
                return cost_result

            return {
                "predicted_month_units": predicted_month_units,
                "avg_daily_kwh": round(avg_per_day, 4),
                "expected_cost": cost_result["estimated_cost"],       # KSEB slab
                "flat_rate_cost": cost_result["flat_rate_cost"],      # Settings rate
            }

        except (ValueError, TypeError):
            return {"error": "Invalid input for current_units or days_elapsed"}


month_forecast_service = MonthForecastService()
