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
from report_generator import create_individual_experiment_report, create_comparative_docx_report
import configparser
import shutil
import json
import copy
from datetime import datetime
import warnings
from jsonschema import validate, ValidationError

# Load configuration and JSON schemas
config = configparser.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config.ini')

if os.path.exists(config_path):
    config.read(config_path)
    GEMINI_API_KEY = config.get('google_ai', 'api_key', fallback='')
    MODEL_ID = config.get('google_ai', 'model_name', fallback='gemini-2.5-pro-preview-06-05')
    GENERATE_CONTENT_API = config.get('google_ai', 'content_api', fallback='generateContent')
else:
    print("Warning: config.ini not found. Please create it with your Google AI configuration.")
    GEMINI_API_KEY = ""
    MODEL_ID = "gemini-2.5-pro-preview-06-05"
    GENERATE_CONTENT_API="generateContent"

# --- Configuration ---
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:{GENERATE_CONTENT_API}?key={GEMINI_API_KEY}"

# --- Helper function to load JSON schemas ---
def load_json_schema(schema_filename: str) -> dict:
    """Loads a JSON schema file from the 'schemas' directory."""
    try:
        # Construct the full path to the schema file
        schema_path = os.path.join(os.getcwd(), 'schemas', schema_filename)
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # --- Add this cleaning step ---
        if '$schema' in schema:
            del schema['$schema']
        # Potentially remove other unsupported keys here in the future if identified
        # print(f"Loaded and cleaned schema from {schema_filename}: {schema_data}") # Optional: for debugging
        # --- End of cleaning step ---
        
        return schema
    except FileNotFoundError:
        print(f"Error: Schema file not found at {schema_path}")
        # Fallback to an empty schema or raise an error, depending on desired behavior
        return {"error": f"Schema file not found: {schema_filename}"}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in schema file: {schema_path}")
        return {"error": f"Invalid JSON in schema file: {schema_filename}"}

# --- Load JSON Schemas ---
PRE_EXPERIMENT_SCHEMA_FROM_FILE = load_json_schema("PRE_EXPERIMENT_SCHEMA.json")
POST_EXPERIMENT_SCHEMA_FROM_FILE = load_json_schema("POST_EXPERIMENT_SCHEMA.json")

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
        # Handle case where all differences are zero for t-test
        if np.all(energy_diff == 0):
            t_stat = 0.0
            p_energy = 1.0
        elif np.var(energy_diff) == 0: # All differences are same non-zero value (highly unlikely for real data)
            # This case would also yield NaN from ttest_1samp due to zero variance.
            # Treat as perfectly predictable deviation if it occurs.
            # For hypothesis: if mean diff is not zero, p_energy should be small.
            # However, ttest_1samp handles this by returning nan.
            # A more robust approach might be needed if this specific edge case is critical.
            # For now, let ttest_1samp produce NaN, which will fail p > alpha.
            # Or, if mean(energy_diff) is exactly 0 (but not all zeros), it's like the np.all case.
            # The primary concern is avoiding NaN from zero variance when diffs are truly all zero.
            t_stat, p_energy = stats.ttest_1samp(energy_diff, 0) # This will likely be nan/nan
        else:
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

# --- Model Calibration ---

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

    # --- TEMPORARY MINIMAL TEST ---
    # test_prompt = "Provide a simple greeting."
    # minimal_schema = {
    #     "type": "OBJECT",
    #     "properties": {
    #         "greeting": {"type": "STRING"}
    #     },
    #     "required": ["greeting"]
    # }
    # Convert the schema keys to camelCase as per documentation for responseSchema
    # This is a guess, the official docs show responseSchema's content itself using camelCase for its own properties like "type", "properties"
    # but the actual schema content for "recipeName" etc. is still user-defined.
    # Let's ensure our schema definition matches standard JSON schema structure first.
    # The field name in the payload should be responseSchema (camelCase).
    
    # print(prompt_text)
    # print(schema)
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt_text}] 
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema 
        }
    }
    # --- END OF TEMPORARY MINIMAL TEST ---


    # print(schema)

    # headers = {'Content-Type': 'application/json'}
    # payload = { "contents": [{
    #         "role": "user",  
    #         "parts": [{"text": prompt_text}]
    #     }],
    #     "generationConfig": {
    #         "responseMimeType": "application/json",
    #         "responseSchema": schema
    #     }
    # }
    
# # "generationConfig": {
#         "responseMimeType": "application/json",
#         "responseSchema": {
#           "type": "ARRAY",
#           "items": {
#             "type": "OBJECT",
#             "properties": {
#               "recipeName": { "type": "STRING" },
#               "ingredients": {
#                 "type": "ARRAY",
#                 "items": { "type": "STRING" }
#               }
#             },
#             "propertyOrdering": ["recipeName", "ingredients"]
#           }
#         }
#       }


    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        response_json = response.json()
        
        # Extract the text content which is a JSON string
        json_string = response_json['candidates'][0]['content']['parts'][0]['text']
        
        # Parse the JSON string into a Python dictionary
        parsed_json = json.loads(json_string)

        # Validate the response against the schema
        if schema and "error" not in schema: # Check if schema is valid before using it
            try:
                validate(instance=parsed_json, schema=schema)
            except ValidationError as e:
                print(f"LLM response validation error: {e.message}")
                # Potentially add more robust error handling or logging here
                # For now, returning parsed_json even if validation fails as per requirement

        return parsed_json

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": str(e)}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing Gemini API response: {e}")
        print(f"Raw Response: {response.text}")
        return {"error": "Failed to parse API response."}

# --- Comparative Report Generation ---
def generate_comparative_report(initial_report_sections, all_scenarios_data, output_base_dir, pre_experiment_schema_content_actual):
    """
    Generates a comparative analysis report using two LLM calls.
    """
    print("\n[PHASE 4] Generating Comparative Analysis Report...")

    comparative_schema_obj = load_json_schema("COMPARATIVE_SCHEMA.json")
    if 'error' in comparative_schema_obj:
        print("Error loading COMPARATIVE_SCHEMA.json. Aborting comparative report generation.")
        return None, None

    comparative_output_dir = os.path.join(output_base_dir, "comparative_report")
    os.makedirs(comparative_output_dir, exist_ok=True)
    print(f"Created comparative report output directory: {comparative_output_dir}")

    # Prepare data for prompts
    introduction_text = initial_report_sections.get('introduction', 'No introduction found in initial sections.')
    scenario_names = list(all_scenarios_data.keys())

    # Data for Prompt 1 (initial structure)
    prompt1_text = f"""
    Overall Introduction from Initial Experiment Design:
    {introduction_text}

    Original Experiment Goals (from experiment_goals.md):
    {pre_experiment_schema_content_actual}

    Task:
    You are tasked with creating an overall introduction for a comparative report that evaluates a scientific model across multiple experimental scenarios.
    The experiments conducted were for the following scenarios: {', '.join(scenario_names)}.

    Based on the provided overall introduction and the experiment goals, please generate:
    1.  `overall_introduction`: A comprehensive introduction suitable for a comparative report that synthesizes the initial experiment's purpose and the multi-scenario validation approach.
    2.  `experiment_summaries`: An initial draft for summarizing each experiment. For each scenario in [{', '.join(scenario_names)}], create an entry with its `scenario_name`. The `key_finding` and `metrics` can be placeholder text for now, as they will be populated in a subsequent step.
    3.  `overall_conclusion`: A preliminary draft for the overall conclusion, anticipating that detailed findings will be integrated later.

    Ensure the output strictly adheres to the COMPARATIVE_SCHEMA.
    """
    print("\nCalling LLM for comparative report - Step 1 (Overall Introduction and Structure)...")
    llm_response_1 = call_gemini_api_with_requests(prompt1_text, comparative_schema_obj)

    if 'error' in llm_response_1:
        print("Error in LLM call 1 for comparative report. Aborting.")
        return None, llm_response_1 # Return None for first, and the error response for second to indicate failure at step 1

    try:
        llm_response_1_path = os.path.join(comparative_output_dir, "comparative_llm_response_1.json")
        with open(llm_response_1_path, 'w') as f:
            json.dump(llm_response_1, f, indent=4)
        print(f"Saved comparative LLM response (Step 1) to {llm_response_1_path}")
    except Exception as e:
        print(f"Error saving comparative_llm_response_1 to JSON: {e}")


    # Prepare detailed data for Prompt 2
    detailed_scenario_summaries = []
    for scenario_name, data in all_scenarios_data.items():
        metrics = data.get('validation_metrics', {})
        hyp_results = data.get('hypothesis_results', {})
        key_finding_proxy = data.get('results_interpretation', {}).get('conclusion_narrative', 'Conclusion not available for this scenario.')

        # Explicitly convert metrics and hypothesis_supported to Python native types
        current_rmse = metrics.get('rmse', 0)
        current_nse = metrics.get('nse', 0)
        is_hypothesis_supported = hyp_results.get('hypothesis_supported', False)

        summary_item = {
            "scenario_name": scenario_name,
            "rmse": float(current_rmse),  # Ensure it's a Python float
            "nse": float(current_nse),    # Ensure it's a Python float
            "hypothesis_supported": bool(is_hypothesis_supported), # Ensure it's a Python bool
            "key_finding": key_finding_proxy
        }
        detailed_scenario_summaries.append(summary_item)

    # Use the overall_introduction from the first LLM call if available
    prompt2_overall_introduction = llm_response_1.get('overall_introduction', introduction_text)

    prompt2_text = f"""
    Overall Introduction (from previous step):
    {prompt2_overall_introduction}

    Detailed Scenario Data:
    {json.dumps(detailed_scenario_summaries, indent=2)}

    Task:
    Based on the previously generated overall introduction and the detailed data from the experimental scenarios provided above, please complete the comparative report.
    Specifically:
    1.  `experiment_summaries`: For each scenario, populate the `key_finding` and `metrics` (rmse, nse, hypothesis_supported) using the detailed data provided.
    2.  `comparative_analysis`: Generate a comprehensive analysis. This should include:
        *   `performance_overview`: A general summary of how the model's performance (RMSE, NSE, hypothesis support) varied across the different scenarios.
        *   `cross_scenario_insights`: Specific insights derived from comparing the outcomes, highlighting any patterns, trade-offs, or unexpected results observed when looking at the scenarios side-by-side.
    3.  `overall_conclusion`: Refine and finalize the overall conclusion for the entire set of experiments, incorporating the synthesized findings and comparative insights.

    Ensure the output strictly adheres to the COMPARATIVE_SCHEMA and integrates information from all provided scenarios.
    """
    print("\nCalling LLM for comparative report - Step 2 (Detailed Analysis and Conclusion)...")
    llm_response_2 = call_gemini_api_with_requests(prompt2_text, comparative_schema_obj)

    if 'error' in llm_response_2:
        print("Error in LLM call 2 for comparative report.")
        # llm_response_1 might be valid, so return it
        return llm_response_1, llm_response_2

    try:
        llm_response_2_path = os.path.join(comparative_output_dir, "comparative_llm_response_2.json")
        with open(llm_response_2_path, 'w') as f:
            json.dump(llm_response_2, f, indent=4)
        print(f"Saved comparative LLM response (Step 2) to {llm_response_2_path}")
    except Exception as e:
        print(f"Error saving comparative_llm_response_2 to JSON: {e}")

    return llm_response_1, llm_response_2


# --- Main Workflow Orchestration ---

def run_scientific_workflow():
    """
    Runs the complete scientific workflow from initial analysis to final report.
    """
    print("="*80)
    print("STARTING SCIENTIFIC WORKFLOW AUTOMATION")
    print("="*80)

    output_base_dir = "output"
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Created base output directory: {output_base_dir}")

    # 1. First LLM Call: Generate initial report sections
    print("\n[PHASE 1] Generating Initial Report Sections...")
    try:
        with open('experiment_goals.md', 'r') as f:
            methodology_prompt = f.read()
        print("Loaded experiment goals from experiment_goals.md")
    except FileNotFoundError:
        print("Error: `experiment_goals.md` not found. Please create it.")
        return

    initial_report_sections = call_gemini_api_with_requests(methodology_prompt, PRE_EXPERIMENT_SCHEMA_FROM_FILE)
    if 'error' in initial_report_sections:
        print("Failed to generate initial report sections. Aborting.")
        return
    print("Successfully generated Abstract, Introduction, Hypothesis, and Methods.")

    initial_sections_path = os.path.join(output_base_dir, "common_initial_sections.json")
    try:
        with open(initial_sections_path, 'w') as f:
            json.dump(initial_report_sections, f, indent=4)
        print(f"Saved initial report sections to {initial_sections_path}")
    except Exception as e:
        print(f"Error saving initial_report_sections to JSON: {e}")


    # 2. Simulation and Analysis for each scenario
    scenarios = ['support', 'fail', 'marginal']
    results_summary = {} # This will still hold data for potential later use or debugging

    for scenario in scenarios:
        print(f"\n{'='*20} RUNNING SCENARIO: {scenario.upper()} {'='*20}")

        scenario_output_dir = os.path.join(output_base_dir, scenario)
        os.makedirs(scenario_output_dir, exist_ok=True)
        print(f"Created scenario output directory: {scenario_output_dir}")
        
        # Load data for the current scenario
        data_path = f"test_data/{scenario}_data.json"
        try:
            with open(data_path, 'r') as f:
                loaded_data = json.load(f)
            print(f"Loaded data from {data_path}")
        except FileNotFoundError:
            print(f"Error: Data file not found for scenario '{scenario}' at {data_path}")
            results_summary[scenario] = {"error": f"Data file not found: {data_path}"}
            continue # Skip to the next scenario
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {data_path}")
            results_summary[scenario] = {"error": f"JSON decode error: {data_path}"}
            continue # Skip to the next scenario

        flow_inputs_list = loaded_data.get("flow_inputs")
        time_steps_list = loaded_data.get("time_steps")
        observed_data_dict_list = loaded_data.get("observed_data")

        if flow_inputs_list is None or time_steps_list is None or observed_data_dict_list is None:
            print(f"Error: Data missing in {data_path}. Required keys: 'flow_inputs', 'time_steps', 'observed_data'.")
            results_summary[scenario] = {"error": f"Data missing in {data_path}"}
            continue

        flow_inputs = np.array(flow_inputs_list)
        time_steps = np.array(time_steps_list)

        # Convert observed_data dictionary lists to numpy arrays or pandas Series
        # For calibrate_model, we need a DataFrame with 'storage_level'
        # For ModelValidator and plotting, we primarily use 'storage_level' as a Series/array.
        observed_data_np = {}
        for key, value in observed_data_dict_list.items():
            observed_data_np[key] = np.array(value)
        
        # This is what calibrate_model expects
        observed_data_for_calibration = pd.DataFrame({'storage_level': observed_data_np['storage_level']})

        print(f"Loaded and processed data for scenario '{scenario}' with {len(time_steps)} time points.")

        # Calibrate model
        calibration_results = calibrate_model(observed_data_for_calibration, flow_inputs, time_steps)
        print(f"Model calibration complete. Final RMSE: {calibration_results['calibration_rmse']:.3f}")
        
        # Simulate with calibrated model
        calibrated_model = SystemModel(
            calibration_results['storage_capacity'],
            calibration_results['overflow_coefficient'],
            calibration_results['storage_efficiency']
        )
        predictions_df = calibrated_model.simulate(flow_inputs, time_steps)
        
        # Validate predictions
        # ModelValidator expects observations for 'storage_level' as a 1D array or Series
        observed_storage_level_for_validation = observed_data_np['storage_level']
        validator = ModelValidator(predictions_df['storage_level'], observed_storage_level_for_validation)
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
        
        results_interpretation = call_gemini_api_with_requests(results_prompt, POST_EXPERIMENT_SCHEMA_FROM_FILE)
        if 'error' in results_interpretation:
            print(f"Warning: Failed to generate interpretation for {scenario} scenario.")
        else:
            print(f"Successfully generated interpretation for {scenario} scenario.")

        # Store all results for this scenario
        scenario_data = {
            'validation_metrics': validation_metrics,
            'hypothesis_results': hypothesis_results,
            'results_interpretation': results_interpretation, # This is the post-experiment LLM response
            'predictions': predictions_df['storage_level'].tolist(),
            'observations': observed_storage_level_for_validation.tolist(),
            'time_steps': time_steps.tolist(),
            'calibration_results': calibration_results,
            # Store the full observed_data_np if other parts are needed later for comprehensive reporting
            'full_observed_data_np': {k: v.tolist() for k, v in observed_data_np.items()}
        }
        results_summary[scenario] = scenario_data

        # Save individual JSON files for the scenario
        pre_experiment_json_path = os.path.join(scenario_output_dir, "pre_experiment_llm_response.json")
        post_experiment_json_path = os.path.join(scenario_output_dir, "post_experiment_llm_response.json")

        try:
            shutil.copy(initial_sections_path, pre_experiment_json_path)
            print(f"Copied common initial sections to {pre_experiment_json_path}")
        except Exception as e:
            print(f"Error copying initial sections for scenario {scenario}: {e}")

        try:
            with open(post_experiment_json_path, 'w') as f:
                json.dump(results_interpretation, f, indent=4)
            print(f"Saved post-experiment LLM response to {post_experiment_json_path}")
        except Exception as e:
            print(f"Error saving results_interpretation for scenario {scenario} to JSON: {e}")

        # 4. Generate individual experiment report for this scenario
        # The actual function `create_individual_experiment_report` will be fully defined in a later subtask.
        # For now, we anticipate its signature and call it.
        # We pass initial_report_sections (pre-LLM) and scenario_data (which includes post-LLM results_interpretation)
        print(f"\n[PHASE 3 - {scenario.upper()}] Compiling Individual Experiment Report...")
        # Placeholder for the new report generator call
        # create_individual_experiment_report(scenario, initial_report_sections, scenario_data, scenario_output_dir)
        # print(f"Report generation called for {scenario} (actual function to be implemented).")
        # For now, to avoid NameError, let's comment out the actual call and just print a message
        # We'll also update the import from report_generator later
        # print(f"Placeholder: Would call create_individual_experiment_report for {scenario} in {scenario_output_dir}")
        # Simulate that the function from report_generator.py is being used, even if it's the old one for now.
        # This will be replaced when report_generator.py is updated.
        # To make the current code runnable without error, we'll call the old function but it won't do what we eventually want.
        # This is a temporary measure.
        create_individual_experiment_report(scenario, initial_report_sections, scenario_data, scenario_output_dir)


    # Overall completion message
    print("\n="*80)
    print("SCIENTIFIC WORKFLOW COMPLETED FOR ALL SCENARIOS")
    print("="*80)

    # 5. Generate Comparative Report (New Step)
    if results_summary and initial_report_sections and 'error' not in initial_report_sections:
        comparative_response_1, comparative_response_2 = generate_comparative_report(
            initial_report_sections,
            results_summary,
            output_base_dir,
            methodology_prompt # This is the content of experiment_goals.md
        )

        if comparative_response_1 and comparative_response_2 and \
           'error' not in comparative_response_1 and 'error' not in comparative_response_2:
            print("\nSuccessfully generated comparative analysis data (2 LLM calls).")
            comparative_report_docx_path = os.path.join(output_base_dir, "comparative_report") # Ensure this path is consistent
            create_comparative_docx_report(comparative_response_1, comparative_response_2, comparative_report_docx_path)
        elif comparative_response_1 and 'error' in comparative_response_1:
            print("\nFailed to generate comparative analysis data due to error in LLM call 1.")
        elif comparative_response_2 and 'error' in comparative_response_2: # Error in second call, first might be okay
            print("\nFailed to generate complete comparative analysis data due to error in LLM call 2.")
            # Optionally, still try to save the first response if it's useful
            # create_comparative_docx_report(comparative_response_1, None, os.path.join(output_base_dir, "comparative_report"))
        else: # Handles cases where one or both responses might be None without an 'error' key if generate_comparative_report returns None
            print("\nFailed to generate complete comparative analysis data. Responses might be missing.")
    else:
        print("\nSkipping comparative report generation due to missing data or errors in initial sections.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    if not GEMINI_API_KEY:
        print("CRITICAL ERROR: The 'GEMINI_API_KEY' environment variable is not set.")
        print("Please set this variable to your Google Gemini API key to run the script.")
    else:
        run_scientific_workflow()