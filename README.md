# EnSight Model (NILM)

EnSight is a Non-Intrusive Load Monitoring (NILM) system designed to disaggregate total household energy consumption into individual appliance usage. This repository contains the training pipelines, API server, and data processing utilities for the model.

## File & Directory Structure

Here is a detailed explanation of the files and directories in this repository:

### Root Directory
- **`config.yaml`**: Central configuration file defining data paths, appliance file mappings, database paths, and training hyperparameters (sequence length, batch size, epochs, etc.).
- **`start_api.py`**: The entry point for the Flask API server. It exposes endpoints for NILM prediction (`/nilm/predict`) and consumption details, serving as the backend for the EnSight application.
- **`train_complete_system.py`**: The main training script. It loads data, creates the LSTM classification model, trains it on the prepared dataset, and saves the best model and configuration.
- **`train_minimal.py`**: A lightweight version of the training script, likely used for quick debugging or testing the training loop with a smaller subset of data.
- **`generate_evaluation_report.py`**: A script to evaluate the trained model's performance. It likely generates metrics (accuracy, F1-score) and reports based on test data.
- **`model_output.py`**: Contains helper functions or classes to handle the output strings or data structures produced by the model predictions.
- **`fix_power_data.py`**: A utility script for preprocessing or fixing issues in the raw power data files before they are used for training.
- **`data_quality_check.py`**: A script to verify the quality and integrity of the input data (e.g., checking for missing values or format errors).
- **`test_db_integration.py`**: Integration tests to verify connections and queries to the SQLite database (`energy_data.db`).
- **`requirements.txt`**: Lists all Python dependencies required to run the project.

### Directories
- **`api/`**: Contains the modularized API logic.
    - `consumptiondetails.py`: Blueprint for handling consumption-related API requests.
    - `nilm_integration.py`: Integration logic for running model predictions within the API.
    - `db_manager.py`: Database connection and management utilities.
- **`src/`**: Source code for the core model architecture and data processing.
    - `models.py` (implied): Definitions of the PyTorch models (e.g., LSTM, CNN).
    - `preprocessor.py` (implied): Data loading, normalization, and sequence creation logic.
- **`models/`**: Directory where the trained model artifacts (e.g., `best_classification_model.pth`) and `training_config.json` are saved during training.
- **`saved_models/`**: Additional directory for storing model checkpoints or alternative versions.
- **`evaluation_results/`**: Stores generated evaluation reports, charts, or metrics files.
- **`plots/`**: Directory for saving visual plots generated during data analysis or model evaluation.
- **`data/`** (referenced in config): Expected location for raw and processed input data files.
- **`venv/`**: (Optional) Python virtual environment directory.

### Data
- **`energy_data.db`**: An SQLite database file used to store energy consumption records or metadata.

## Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model:**
    ```bash
    python train_complete_system.py
    ```

3.  **Start the API Server:**
    ```bash
    python start_api.py
    ```
    The API will be available at `http://localhost:5000`.
