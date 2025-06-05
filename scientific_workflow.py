import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
import json
import requests
from report_generator import create_scientific_report
import configparser
import os
import json
import copy
from datetime import datetime
import warnings

# Load configuration and JSON schemas
config = configparser.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config.ini')

if os.path.exists(config_path):
    config.read(config_path)
    GEMINI_API_KEY = config.get('google_ai', 'api_key', fallback='')
    MODEL_ID = config.get('google_ai', 'model_name', fallback='gemini-2.5-pro-preview-05-06')
    GENERATE_CONTENT_API = config.get('google_ai', 'content_api', fallback='generateContent')
else:
    print("Warning: config.ini not found. Please create it with your Google AI configuration.")
    GEMINI_API_KEY = ""
    MODEL_ID = "gemini-2.5-pro-preview-05-06"
    GENERATE_CONTENT_API="generateContent"

# --- Configuration ---
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:${GENERATE_CONTENT_API}?key=${GEMINI_API_KEY}"

# --- JSON Schemas for Structured LLM Output ---

# Schema for the first LLM call (pre-experiment)
PRE_EXPERIMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "abstract": {
            "type": "STRING",
            "description": "A concise summary of the entire study, including aims, methods, key findings, and conclusions."
        },
        "introduction": {
            "type": "STRING",
            "description": "Background on Energy Systems, the importance of model validation, and the study's objectives."
        },
        "hypothesis": {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING", "description": "The formal null and alternative hypotheses."},
                "independent_variable": {"type": "STRING"},
                "dependent_variable": {"type": "STRING"}
            }
        },
        "methods": {
            "type": "STRING",
            "description": "A detailed description of the experimental design, including the model used, data generation, statistical tests, and validation criteria."
        }
    },
    "required": ["abstract", "introduction", "hypothesis", "methods"]
}

# Schema for the second LLM call (post-experiment interpretation)
POST_EXPERIMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "results": {
            "type": "OBJECT",
            "properties": {
                 "summary": {"type": "STRING", "description": "A summary of the key statistical findings and model performance metrics."},
                 "data_interpretation": {"type": "STRING", "description": "Interpretation of the statistical data table provided in the prompt."}
            }
        },
        "discussion": {
            "type": "STRING",
            "description": "Discusses the implications of the results, compares them to the hypothesis, explains why the model performed as it did, and addresses limitations."
        },
        "conclusion": {
            "type": "STRING",
            "description": "A final summary of the study's findings and their significance for the model's practical utility."
        }
    },
    "required": ["results", "discussion", "conclusion"]
}


# --- Core Model and Validation Classes (from starter) ---

class SystemModel:
    def __init__(self, storage_capacity=100, overflow_coefficient=0.8, storage_efficiency=0.9):
        self.storage_capacity = storage_capacity
        self.overflow_coefficient = overflow_coefficient
        self.storage_efficiency = storage_efficiency
        self.current_storage = 0
        
    def step(self, flow_input, dt=1.0):
        energy_in = flow_input * self.storage_efficiency * dt
        potential_storage = self.current_storage + energy_in
        if potential_storage > self.storage_capacity:
            overflow = (potential_storage - self.storage_capacity) * self.overflow_coefficient
            self.current_storage = self.storage_capacity
        else:
            overflow = 0
            self.current_storage = potential_storage
        storage_loss = self.current_storage * 0.05 * dt
        self.current_storage = max(0, self.current_storage - storage_loss)
        return {
            'storage_level': self.current_storage, 'overflow': overflow,
            'energy_input': flow_input, 'storage_loss': storage_loss
        }
    
    def simulate(self, flow_inputs, time_steps):
        results = []
        self.current_storage = 0
        for i, flow_input in enumerate(flow_inputs):
            result = self.step(flow_input)
            result['time'] = time_steps[i]
            results.append(result)
        return pd.DataFrame(results)

class ModelValidator:
    def __init__(self, predictions, observations):
        self.predictions = predictions
        self.observations = observations
        self.validation_results = {}
        
    def calculate_metrics(self):
        rmse = np.sqrt(mean_squared_error(self.observations, self.predictions))
        mae = mean_absolute_error(self.observations, self.predictions)
        mape_values = []
        for obs, pred in zip(self.observations, self.predictions):
            if abs(obs) > 1e-6:
                mape_values.append(abs((obs - pred) / obs))
        mape = np.mean(mape_values) * 100 if mape_values else 0
        r_pearson, p_pearson = stats.pearsonr(self.observations, self.predictions)
        r_spearman, p_spearman = stats.spearmanr(self.observations, self.predictions)
        nse = 1 - (np.sum((self.observations - self.predictions)**2) / 
                  np.sum((self.observations - np.mean(self.observations))**2))
        energy_diff = self.observations - self.predictions
        t_stat, p_energy = stats.ttest_1samp(energy_diff, 0)
        self.validation_results = {
            'rmse': rmse, 'mae': mae, 'mape': mape, 'pearson_r': r_pearson,
            'pearson_p': p_pearson, 'spearman_r': r_spearman, 'spearman_p': p_spearman,
            'nse': nse, 'energy_balance_t': t_stat, 'energy_balance_p': p_energy
        }
        return self.validation_results
    
    def hypothesis_test(self, alpha=0.05):
        if not self.validation_results:
            self.calculate_metrics()
        validation_criteria = {
            'energy_conservation': self.validation_results['energy_balance_p'] > alpha,
            'good_correlation': self.validation_results['pearson_p'] < alpha and abs(self.validation_results['pearson_r']) > 0.7,
            'acceptable_nse': self.validation_results['nse'] > 0.5,
            'reasonable_error': self.validation_results['mape'] < 20
        }
        criteria_passed = sum(validation_criteria.values())
        hypothesis_supported = criteria_passed >= 3
        return {
            'hypothesis_supported': hypothesis_supported,
            'criteria_passed': criteria_passed,
            'total_criteria': len(validation_criteria),
            'detailed_results': validation_criteria
        }

# --- Data Generation and Model Calibration ---

def generate_synthetic_data(scenario='support'):
    time_steps = np.arange(0, 100, 1)
    base_flow = 20 + 10 * np.sin(time_steps * 0.1) + np.random.normal(0, 2, len(time_steps))
    base_flow = np.maximum(base_flow, 0)
    model = SystemModel(120, 0.75, 0.85)
    true_data = model.simulate(base_flow, time_steps)
    if scenario == 'support':
        true_data['storage_level'] += np.random.normal(0, 2, len(true_data))
    elif scenario == 'fail':
        true_data['storage_level'] = true_data['storage_level'] * 1.5 + 20 + np.random.normal(0, 10, len(true_data))
    elif scenario == 'marginal':
        true_data['storage_level'] = true_data['storage_level'] * 1.2 + 10 + np.random.normal(0, 5, len(true_data))
    return base_flow, time_steps, true_data

def calibrate_model(observed_data, flow_inputs, time_steps):
    def objective_function(params):
        storage_capacity, overflow_coeff, storage_eff = params
        if not (10 <= storage_capacity <= 500 and 0.1 <= overflow_coeff <= 1.0 and 0.1 <= storage_eff <= 1.0):
            return 1e6
        model = SystemModel(storage_capacity, overflow_coeff, storage_eff)
        predictions = model.simulate(flow_inputs, time_steps)
        rmse = np.sqrt(mean_squared_error(observed_data['storage_level'], predictions['storage_level']))
        return rmse
    initial_params = [100, 0.8, 0.9]
    bounds = [(10, 500), (0.1, 1.0), (0.1, 1.0)]
    result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
    optimal_params = result.x
    return {
        'storage_capacity': optimal_params[0],
        'overflow_coefficient': optimal_params[1], 
        'storage_efficiency': optimal_params[2],
        'calibration_rmse': result.fun
    }

# --- LLM Communication Logic ---

def call_gemini_api_with_requests(prompt_text, schema):
    """
    Calls the Gemini API using the requests module with a specified JSON schema.
    """
    if not GEMINI_API_KEY:
        error_message = "Google API key not configured. Please set the GEMINI_API_KEY environment variable."
        print(f"Warning: {error_message}")
        # Return a dummy JSON structure to prevent crashes
        return {"error": error_message}

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": schema
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        response_json = response.json()
        
        # Extract the text content which is a JSON string
        json_string = response_json['candidates'][0]['content']['parts'][0]['text']
        
        # Parse the JSON string into a Python dictionary
        parsed_json = json.loads(json_string)
        return parsed_json

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": str(e)}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing Gemini API response: {e}")
        print(f"Raw Response: {response.text}")
        return {"error": "Failed to parse API response."}


# --- Main Workflow Orchestration ---

def run_scientific_workflow():
    """
    Runs the complete scientific workflow from initial analysis to final report.
    """
    print("="*80)
    print("STARTING SCIENTIFIC WORKFLOW AUTOMATION")
    print("="*80)

    # 1. First LLM Call: Generate initial report sections
    print("\n[PHASE 1] Generating Initial Report Sections...")
    try:
        with open('experiment_goals.md', 'r') as f:
            methodology_prompt = f.read()
        print("Loaded experiment goals from experiment_goals.md")
    except FileNotFoundError:
        print("Error: `experiment_goals.md` not found. Please create it.")
        return

    initial_report_sections = call_gemini_api_with_requests(methodology_prompt, PRE_EXPERIMENT_SCHEMA)
    if 'error' in initial_report_sections:
        print("Failed to generate initial report sections. Aborting.")
        return
    print("Successfully generated Abstract, Introduction, Hypothesis, and Methods.")

    # 2. Simulation and Analysis for each scenario
    scenarios = ['support', 'fail', 'marginal']
    results_summary = {}

    for scenario in scenarios:
        print(f"\n{'='*20} RUNNING SCENARIO: {scenario.upper()} {'='*20}")
        
        # Generate data
        flow_inputs, time_steps, observed_data = generate_synthetic_data(scenario)
        print(f"Generated synthetic data with {len(time_steps)} time points.")
        
        # Calibrate model
        calibration_results = calibrate_model(observed_data, flow_inputs, time_steps)
        print(f"Model calibration complete. Final RMSE: {calibration_results['calibration_rmse']:.3f}")
        
        # Simulate with calibrated model
        calibrated_model = SystemModel(
            calibration_results['storage_capacity'],
            calibration_results['overflow_coefficient'],
            calibration_results['storage_efficiency']
        )
        predictions_df = calibrated_model.simulate(flow_inputs, time_steps)
        
        # Validate predictions
        validator = ModelValidator(predictions_df['storage_level'], observed_data['storage_level'])
        validation_metrics = validator.calculate_metrics()
        hypothesis_results = validator.hypothesis_test()
        
        print("\nValidation Metrics:")
        print(f"  - RMSE: {validation_metrics['rmse']:.3f}, NSE: {validation_metrics['nse']:.3f}")
        print(f"  - Hypothesis Supported: {hypothesis_results['hypothesis_supported']} ({hypothesis_results['criteria_passed']}/{hypothesis_results['total_criteria']} criteria passed)")

        # 3. Second LLM Call: Interpret the results for this scenario
        print("\n[PHASE 2] Generating Interpretation for Scenario Results...")
        results_prompt = f"""
        Interpret the following scientific model validation results for the '{scenario}' scenario.
        
        Provide a detailed interpretation in the RESULTS, DISCUSSION, and CONCLUSION sections.
        
        Statistical Data:
        - RMSE: {validation_metrics['rmse']:.3f}
        - MAE: {validation_metrics['mae']:.3f}
        - MAPE: {validation_metrics['mape']:.2f}%
        - Nash-Sutcliffe Efficiency (NSE): {validation_metrics['nse']:.3f}
        - Pearson Correlation (r): {validation_metrics['pearson_r']:.3f}
        - Pearson p-value: {validation_metrics['pearson_p']:.4f}
        - Hypothesis Test Result: The hypothesis that the model is a good fit was {'SUPPORTED' if hypothesis_results['hypothesis_supported'] else 'REJECTED'}.
        - Criteria Passed: {hypothesis_results['criteria_passed']} out of {hypothesis_results['total_criteria']}.
        
        Based on this data, generate the JSON output.
        """
        
        results_interpretation = call_gemini_api_with_requests(results_prompt, POST_EXPERIMENT_SCHEMA)
        if 'error' in results_interpretation:
            print(f"Warning: Failed to generate interpretation for {scenario} scenario.")
        else:
            print(f"Successfully generated interpretation for {scenario} scenario.")

        # Store all results for final report generation
        results_summary[scenario] = {
            'validation_metrics': validation_metrics,
            'hypothesis_results': hypothesis_results,
            'results_interpretation': results_interpretation,
            'predictions': predictions_df['storage_level'].values,
            'observations': observed_data['storage_level'].values,
            'time_steps': time_steps
        }

    # 4. Final Step: Compile the full scientific report
    print("\n[PHASE 3] Compiling Final Scientific Report...")
    if results_summary:
        create_scientific_report(results_summary, initial_report_sections)
    else:
        print("No results were generated, skipping report creation.")

    print("\n="*80)
    print("SCIENTIFIC WORKFLOW COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    if not GEMINI_API_KEY:
        print("CRITICAL ERROR: The 'GEMINI_API_KEY' environment variable is not set.")
        print("Please set this variable to your Google Gemini API key to run the script.")
    else:
        run_scientific_workflow()