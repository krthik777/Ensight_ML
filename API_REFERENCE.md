# EnSight ML API — Request & Response Reference

> **For Node.js Backend Developers**  
> Base URL: `http://<ml-server>:5050`  
> All requests use `Content-Type: application/json`

---

## How It Works

The ML backend connects directly to the shared MongoDB database.  
You only need to send **identifiers** — the ML backend fetches supporting data itself.

| What you send | What the ML backend fetches automatically |
|---|---|
| `userId` + `roomId` | Last 50 power readings from `powerreadings` |
| `userId` | `settings.budget.ratePerKwh`, `budget.monthly`, `budget.currency` |
| `roomId` | `rooms.threshold` (for anomaly detection) |
| *(computed internally)* | `days_elapsed` from current date (no need to send) |

---

## Endpoints

---

### 1. `POST /detect-appliances`

Detects which appliances are currently ON based on recent power readings.  
Results are **automatically saved** to the `appliancedetections` collection.

**Required Body**
```json
{
  "userId": "<MongoDB ObjectId string>",
  "roomId": "<MongoDB ObjectId string>"
}
```

| Field | Type | Required | Source |
|---|---|---|---|
| `userId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `roomId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `mains_sequence` | Array of Numbers | ❌ No | Auto-fetched from `powerreadings.power` (last 50) |

**Example Request**
```json
POST /detect-appliances
{
  "userId": "65f1a2b3c4d5e6f7a8b9c0d1",
  "roomId": "65f1a2b3c4d5e6f7a8b9c0d2"
}
```

**Example Response**
```json
{
  "success": true,
  "active_appliances": [
    "air_conditioner",
    "kitchen_outlets",
    "television",
    "washing_machine"
  ],
  "confidence": {
    "air_conditioner": 0.98,
    "kitchen_outlets": 0.96,
    "television": 0.99,
    "washing_machine": 0.98
  }
}
```

**Error Response (no readings in DB)**
```json
{
  "success": false,
  "error": "Provide at least 'roomId' (or 'mains_sequence' for testing)"
}
```

---

### 2. `POST /predict-appliance-power`

Returns the estimated power consumption (in Watts) for each individual appliance.

**Required Body**
```json
{
  "userId": "<MongoDB ObjectId string>",
  "roomId": "<MongoDB ObjectId string>"
}
```

| Field | Type | Required | Source |
|---|---|---|---|
| `userId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `roomId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `mains_sequence` | Array of Numbers | ❌ No | Auto-fetched from `powerreadings.power` (last 50) |

**Example Request**
```json
POST /predict-appliance-power
{
  "userId": "65f1a2b3c4d5e6f7a8b9c0d1",
  "roomId": "65f1a2b3c4d5e6f7a8b9c0d2"
}
```

**Example Response**
```json
{
  "success": true,
  "appliances": {
    "air_conditioner": 1667.91,
    "fridge": 0,
    "iron": 0,
    "kitchen_outlets": 161.49,
    "laptop_computer": 0,
    "television": 73.78,
    "washing_machine": 148.74,
    "water_filter": 0,
    "water_motor": 0,
    "other": 34.12
  }
}
```

> `other` = unaccounted power (mains total − sum of known appliances)  
> Values are in **Watts (W)**. `0` means the appliance is OFF.

---

### 3. `POST /predict-cost`

Estimates the monthly electricity bill using the KSEB 2025-2027 slab tariff.  
Daily usage is calculated from the `powerreadings.energy` cumulative delta over the last 24 hours.  
The rate and budget are read from the user's `settings` document.

**Required Body**
```json
{
  "userId": "<MongoDB ObjectId string>",
  "roomId": "<MongoDB ObjectId string>"
}
```

| Field | Type | Required | Source |
|---|---|---|---|
| `userId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `roomId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `daily_usage_kwh` | Number | ❌ No | Auto-calculated from `powerreadings.energy` (last 24h delta) |

**Example Request**
```json
POST /predict-cost
{
  "userId": "65f1a2b3c4d5e6f7a8b9c0d1",
  "roomId": "65f1a2b3c4d5e6f7a8b9c0d2"
}
```

**Example Response**
```json
{
  "success": true,
  "estimated_monthly_units": 246.0,
  "estimated_cost": 1137.58,
  "flat_rate_cost": 1599.0,
  "currency": "INR",
  "rate_per_kwh_used": 6.5,
  "monthly_budget_kwh": 400,
  "budget_headroom_kwh": 154.0,
  "over_budget": false
}
```

| Response Field | Description |
|---|---|
| `estimated_monthly_units` | Projected kWh for the full month |
| `estimated_cost` | Bill via KSEB telescopic slabs (primary) |
| `flat_rate_cost` | Bill via `settings.budget.ratePerKwh` (secondary) |
| `currency` | From `settings.budget.currency` |
| `rate_per_kwh_used` | From `settings.budget.ratePerKwh` |
| `monthly_budget_kwh` | From `settings.budget.monthly` |
| `budget_headroom_kwh` | `monthly_budget_kwh − estimated_monthly_units` |
| `over_budget` | `true` if projected usage exceeds budget |

**Error Response (no 24h readings found)**
```json
{
  "success": false,
  "error": "No power readings found for this room in the last 24 hours. Ensure the sensor is streaming data."
}
```

---

### 4. `POST /predict-month-end`

Forecasts end-of-month electricity usage and cost using linear extrapolation.  
`current_units` and `days_elapsed` are both derived automatically from MongoDB — no need to send them.

**Required Body**
```json
{
  "userId": "<MongoDB ObjectId string>",
  "roomId": "<MongoDB ObjectId string>"
}
```

| Field | Type | Required | Source |
|---|---|---|---|
| `userId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `roomId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `current_units` | Number | ❌ No | Auto-calculated: `powerreadings.energy` delta since 1st of month |
| `days_elapsed` | Number | ❌ No | Auto-calculated from current date |

**Example Request**
```json
POST /predict-month-end
{
  "userId": "65f1a2b3c4d5e6f7a8b9c0d1",
  "roomId": "65f1a2b3c4d5e6f7a8b9c0d2"
}
```

**Example Response**
```json
{
  "success": true,
  "predicted_month_units": 240.0,
  "avg_daily_kwh": 8.0,
  "expected_cost": 1101.70,
  "flat_rate_cost": 1560.0,
  "currency": "INR",
  "rate_per_kwh_used": 6.5,
  "monthly_budget_kwh": 400,
  "budget_headroom_kwh": 160.0,
  "over_budget": false,
  "days_elapsed": 15,
  "days_remaining": 15
}
```

| Response Field | Description |
|---|---|
| `predicted_month_units` | Projected kWh for full 30-day month |
| `avg_daily_kwh` | Current average daily consumption |
| `expected_cost` | Projected bill via KSEB slabs |
| `flat_rate_cost` | Projected bill via `settings.budget.ratePerKwh` |
| `days_elapsed` | Days since 1st of the current month |
| `days_remaining` | Days left in the month |
| `over_budget` | `true` if forecast exceeds `settings.budget.monthly` |

**Error Response (insufficient monthly data)**
```json
{
  "success": false,
  "error": "No monthly power data found for this room. Ensure the sensor has been streaming data since the 1st of the month."
}
```

---

### 5. `POST /detect-anomaly`

Detects unusual power spikes using two parallel checks:
- **Statistical:** `max > mean + 3 × std_dev`
- **Threshold:** `max > rooms.threshold` (fetched from MongoDB)

**Required Body**
```json
{
  "userId": "<MongoDB ObjectId string>",
  "roomId": "<MongoDB ObjectId string>"
}
```

| Field | Type | Required | Source |
|---|---|---|---|
| `userId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `roomId` | String (ObjectId) | ✅ Yes | Sent from Node.js |
| `mains_sequence` | Array of Numbers | ❌ No | Auto-fetched from `powerreadings.power` (last 50) |

**Example Request**
```json
POST /detect-anomaly
{
  "userId": "65f1a2b3c4d5e6f7a8b9c0d1",
  "roomId": "65f1a2b3c4d5e6f7a8b9c0d2"
}
```

**Example Response — Anomaly Detected**
```json
{
  "success": true,
  "possible_faulty_appliance": true,
  "room_threshold_used": 2000.0,
  "details": {
    "max_power": 3500.0,
    "mean_power": 168.0,
    "statistical_threshold": 1596.0,
    "room_power_threshold": 2000.0,
    "triggered_by": "both"
  }
}
```

**Example Response — No Anomaly**
```json
{
  "success": true,
  "possible_faulty_appliance": false,
  "room_threshold_used": 2000.0
}
```

| `triggered_by` value | Meaning |
|---|---|
| `"statistical"` | Spike detected by 3-sigma rule only |
| `"room_threshold"` | Reading exceeded `rooms.threshold` only |
| `"both"` | Both checks triggered |

---

### 6. `GET /health`

Simple health check — no body required.

**Example Response**
```json
{
  "status": "healthy",
  "service": "EnSight Data Inference API"
}
```

---

## Summary Table

| Endpoint | Must Send | Never Send (auto from DB) |
|---|---|---|
| `POST /detect-appliances` | `userId`, `roomId` | `mains_sequence` |
| `POST /predict-appliance-power` | `userId`, `roomId` | `mains_sequence` |
| `POST /predict-cost` | `userId`, `roomId` | `daily_usage_kwh`, `ratePerKwh`, budget |
| `POST /predict-month-end` | `userId`, `roomId` | `current_units`, `days_elapsed`, `ratePerKwh`, budget |
| `POST /detect-anomaly` | `userId`, `roomId` | `mains_sequence`, `room_threshold` |
| `GET /health` | *(nothing)* | — |

---

## MongoDB Collections Used Per Endpoint

| Collection | Used By |
|---|---|
| `powerreadings.power` | `/detect-appliances`, `/predict-appliance-power`, `/detect-anomaly` |
| `powerreadings.energy` | `/predict-cost`, `/predict-month-end` |
| `rooms.threshold` | `/detect-anomaly` |
| `settings.budget.*` | `/predict-cost`, `/predict-month-end` |
| `appliancedetections` | Written by `/detect-appliances` (auto) |

---

## Override Fields (Testing / Debugging Only)

These fields can be passed to **bypass MongoDB** and use direct values instead.  
Useful during development or when the sensor is offline.

```json
POST /detect-appliances
{
  "userId": "...",
  "roomId": "...",
  "mains_sequence": [1200, 1250, 1300, ...]
}

POST /predict-cost
{
  "userId": "...",
  "roomId": "...",
  "daily_usage_kwh": 8.2
}

POST /predict-month-end
{
  "userId": "...",
  "roomId": "...",
  "current_units": 120,
  "days_elapsed": 15
}
```

> **Note:** If MongoDB is unreachable and override fields are not provided,
> the endpoint returns a `404` with a descriptive error message.
