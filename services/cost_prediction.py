"""
services/cost_prediction.py
-----------------------------
KSEB slab-based electricity cost calculator.

The rate_per_kwh from the user's settings is used as a simple flat-rate
fallback when a caller passes it. The primary calculation always uses the
KSEB 2025-2027 telescopic slab structure.
"""


class CostPredictionService:

    @staticmethod
    def calculate_monthly_cost(daily_usage_kwh, rate_per_kwh: float = 6.5):
        """
        Predict monthly electricity cost using KSEB 2025-2027 slab tariff.

        Args:
            daily_usage_kwh : Average daily consumption in kWh.
            rate_per_kwh    : Flat rate from user settings (INR/kWh).
                             Used to compute a simple parallel estimate;
                             the KSEB slab figure is always the primary result.
        """
        try:
            daily_usage_kwh = float(daily_usage_kwh)
            estimated_monthly_units = daily_usage_kwh * 30

            # ── KSEB 2025-2027 Telescopic Slab Calculation ──────────────
            cost = 0.0
            units = estimated_monthly_units

            if units > 500:
                cost += (units - 500) * 7.10
                units = 500
            if units > 300:
                cost += (units - 300) * 6.50
                units = 300
            if units > 200:
                cost += (units - 200) * 5.20
                units = 200
            if units > 100:
                cost += (units - 100) * 4.00
                units = 100
            if units > 0:
                cost += units * 3.30

            cost += 20          # Fixed charge
            cost *= 1.15        # 15% electricity duty

            # ── Flat-rate parallel estimate using user's settings ────────
            flat_rate_cost = round(estimated_monthly_units * rate_per_kwh, 2)

            return {
                "estimated_monthly_units": round(estimated_monthly_units, 2),
                "estimated_cost": round(cost, 2),           # KSEB slab (primary)
                "flat_rate_cost": flat_rate_cost,           # Simple rate from settings
                "currency": "INR",
            }

        except (ValueError, TypeError):
            return {"error": "Invalid input for daily_usage_kwh"}


cost_prediction_service = CostPredictionService()
