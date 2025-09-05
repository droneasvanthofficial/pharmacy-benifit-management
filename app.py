from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Add this for frontend-backend communication
import pandas as pd
import pickle
from io import StringIO
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load ML model for drug utilization trend
try:
    with open('drug_utilization_trend_model.pkl', 'rb') as f:
        trend_model = pickle.load(f)
    print("✅ Drug utilization trend model loaded")
except Exception as e:
    trend_model = None
    print("❌ Error loading trend model:", e)

# Load therapeutic equivalence data
try:
    with open('theurapatic.pkl', 'rb') as f:
        te_data = pickle.load(f)
    
    # Convert to DataFrame if it's a dictionary
    if isinstance(te_data, dict):
        te_df = pd.DataFrame(te_data)
    else:
        te_df = te_data
    
    print("✅ Therapeutic equivalence data loaded")
    print("Available columns:", te_df.columns.tolist())
except Exception as e:
    te_df = None
    print("❌ Error loading therapeutic equivalence data:", e)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    """
    Unified upload endpoint:
    1. Predict drug utilization trend
    2. Merge with therapeutic equivalence data
    3. Calculate 12% cost reduction analysis
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read CSV from uploaded file
        csv_data = StringIO(file.read().decode('utf-8'))
        df = pd.read_csv(csv_data)
        results = {}

        # --- 1. Drug Utilization Trend Prediction ---
        if trend_model is not None:
            # Check for required columns
            if not all(col in df.columns for col in ['Medication', 'Billing Amount']):
                results['trend_prediction'] = "Warning: Missing 'Medication' or 'Billing Amount' for trend analysis."
            else:
                # Create a copy and clean the data
                temp_df = df[['Medication', 'Billing Amount']].copy()
                temp_df['Billing Amount'] = pd.to_numeric(temp_df['Billing Amount'], errors='coerce')
                temp_df = temp_df.dropna()

                if len(temp_df) > 0:
                    # Simulate time-series features for prediction (since we don't have dates)
                    # This is a placeholder. Your real model would use proper time series data.
                    temp_df['Feature_1'] = np.random.rand(len(temp_df))  # Simulated feature
                    temp_df['Feature_2'] = np.random.rand(len(temp_df))  # Simulated feature

                    # Predict using the model
                    X = temp_df[['Feature_1', 'Feature_2']]
                    predictions = trend_model.predict(X)

                    # ********** THE CRITICAL FIX: Map 0/1 to Clear Labels **********
                    trend_map = {0: 'Stable/Decrease', 1: 'Increase Expected'}
                    temp_df['Trend_Prediction'] = [trend_map[p] for p in predictions.astype(int)]
                    # ***************************************************************

                    # Format the results
                    trend_results = []
                    for _, row in temp_df.iterrows():
                        trend_results.append({
                            "MEDICATION": row['Medication'],
                            "BILLING_AMOUNT": row['Billing Amount'],
                            "TREND_PREDICTION": row['Trend_Prediction']  # This now has the clear label
                        })

                    results['trend_prediction'] = trend_results
                else:
                    results['trend_prediction'] = "No valid data for trend prediction."

        # --- 2. Therapeutic Equivalence & Cost Reduction ---
        if te_df is not None and not df.empty:
            # Standardize column names for merging
            user_drug_col = 'Medication'
            te_drug_col = 'Drug_Name'  # Adjust this to match your theurapatic.pkl column name

            if user_drug_col in df.columns and te_drug_col in te_df.columns:
                # Merge the datasets
                merged_df = pd.merge(df, te_df, left_on=user_drug_col, right_on=te_drug_col, how='left')
                
                # Fill missing values
                columns_to_fill = ['Alternative_Drug', 'Category', 'Alt_Cost', 'Savings_Rx']
                for col in columns_to_fill:
                    if col in merged_df.columns:
                        merged_df[col] = merged_df[col].fillna('N/A')
                
                # Convert results to dictionary
                results['therapeutic_equivalence'] = merged_df.to_dict(orient='records')
            else:
                results['therapeutic_equivalence'] = "Could not merge data. Column names may not match."
        else:
            results['therapeutic_equivalence'] = "Therapeutic equivalence data not available."

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)