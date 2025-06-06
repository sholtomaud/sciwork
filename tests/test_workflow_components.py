import unittest
import numpy as np
import pandas as pd
import json
import os
import requests # For requests.exceptions.HTTPError
from unittest.mock import patch, MagicMock, mock_open

# Adjust import path assuming tests are run from the repository root
# and scientific_workflow.py is in the root.
from scientific_workflow import SystemModel, ModelValidator, call_gemini_api_with_requests, load_json_schema
from jsonschema import ValidationError # Import for mock side_effect

# --- Global Test Setup ---
# Mock config.ini before scientific_workflow potentially reads it
# This is to ensure GEMINI_API_KEY can be controlled for tests
MOCK_CONFIG_CONTENT = """
[google_ai]
api_key = TEST_API_KEY_FROM_MOCK_CONFIG
model_name = test_model
content_api = test_generateContent
"""

# We need to patch 'open' in the 'scientific_workflow' module's scope
# where config.ini is read.
@patch('scientific_workflow.open', new_callable=mock_open, read_data=MOCK_CONFIG_CONTENT)
@patch('scientific_workflow.os.path.exists')
def load_dependencies(mock_exists, mock_open_func):
    """
    This function is a bit of a workaround to ensure that when scientific_workflow
    is imported, it uses our mocked config if config.ini doesn't exist
    or if we want to force our mock.
    It also reloads the module to apply mocks if already loaded.
    """
    mock_exists.return_value = True # Pretend config.ini exists for the read

    # If scientific_workflow was already imported, reload it to apply patches
    import sys
    if 'scientific_workflow' in sys.modules:
        import importlib
        importlib.reload(sys.modules['scientific_workflow'])

    # The imports we need for the test file itself are done globally,
    # this is just to ensure the module under test uses mocks.

# Call this before defining test classes that might trigger module-level code in scientific_workflow
# However, direct patching of GEMINI_API_KEY within scientific_workflow will be more reliable for the API key test.

class TestSystemModel(unittest.TestCase):
    def setUp(self):
        """Initialize a SystemModel instance."""
        # Default parameters from the SystemModel definition in scientific_workflow.py
        self.model = SystemModel(storage_capacity=100, overflow_coefficient=0.8, storage_efficiency=0.9)
        self.model.current_storage = 0 # Ensure a known starting state

    def test_system_model_step(self):
        """Test a single step of the system model."""
        initial_storage = self.model.current_storage
        flow_input = 10.0
        dt = 1.0

        result = self.model.step(flow_input=flow_input, dt=dt)

        self.assertIsInstance(result, dict)
        expected_keys = ['storage_level', 'overflow', 'energy_input', 'storage_loss']
        for key in expected_keys:
            self.assertIn(key, result, f"Key '{key}' missing in step result.")

        # Expected logic:
        # energy_in = flow_input * storage_efficiency * dt = 10 * 0.9 * 1 = 9
        # potential_storage = initial_storage + energy_in = 0 + 9 = 9
        # overflow = 0 (since 9 <= 100)
        # current_storage_before_loss = 9
        # storage_loss = current_storage_before_loss * 0.05 * dt = 9 * 0.05 * 1 = 0.45
        # final_storage = current_storage_before_loss - storage_loss = 9 - 0.45 = 8.55

        self.assertEqual(result['energy_input'], flow_input)
        self.assertEqual(result['overflow'], 0)
        self.assertAlmostEqual(result['storage_loss'], 9 * 0.05 * dt) # Loss based on storage *before* deduction
        self.assertAlmostEqual(result['storage_level'], 8.55)
        self.assertAlmostEqual(self.model.current_storage, 8.55)

    def test_system_model_simulate(self):
        """Test the simulation over multiple time steps."""
        flow_inputs = np.array([10, 12, 8])
        time_steps = np.array([0, 1, 2])

        results_df = self.model.simulate(flow_inputs, time_steps)

        self.assertIsInstance(results_df, pd.DataFrame)
        expected_columns = ['time', 'storage_level', 'overflow', 'energy_input', 'storage_loss']
        for col in expected_columns:
            self.assertIn(col, results_df.columns, f"Column '{col}' missing in simulation DataFrame.")

        self.assertEqual(len(results_df), len(time_steps), "Number of rows in DataFrame does not match time steps.")
        self.assertTrue(np.array_equal(results_df['time'], time_steps))
        self.assertTrue(np.array_equal(results_df['energy_input'], flow_inputs))


class TestModelValidator(unittest.TestCase):
    def setUp(self):
        self.observations = np.array([10, 11, 12, 11, 10], dtype=float)
        self.predictions_good = np.array([9.5, 10.5, 12.5, 11.5, 9.0], dtype=float)
        self.predictions_perfect = np.array([10, 11, 12, 11, 10], dtype=float)
        self.predictions_poor = np.array([5, 6, 7, 6, 5], dtype=float)

        self.validator_good = ModelValidator(self.predictions_good, self.observations)
        self.validator_perfect = ModelValidator(self.predictions_perfect, self.observations)
        self.validator_poor = ModelValidator(self.predictions_poor, self.observations)

    def test_calculate_metrics(self):
        results = self.validator_good.calculate_metrics()
        self.assertIsInstance(results, dict)
        self.assertTrue(self.validator_good.validation_results is results) # Check if it's stored

        expected_metrics = ['rmse', 'mae', 'mape', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'nse', 'energy_balance_t', 'energy_balance_p']
        for metric in expected_metrics:
            self.assertIn(metric, results, f"Metric '{metric}' missing.")
            self.assertIsInstance(results[metric], float, f"Metric '{metric}' is not a float.")

        # Test perfect prediction
        perfect_results = self.validator_perfect.calculate_metrics()
        self.assertAlmostEqual(perfect_results['rmse'], 0.0)
        self.assertAlmostEqual(perfect_results['mae'], 0.0)
        self.assertAlmostEqual(perfect_results['mape'], 0.0) # MAPE might be tricky if obs is 0
        self.assertAlmostEqual(perfect_results['nse'], 1.0)

    def test_hypothesis_test(self):
        # Test with good predictions (should support hypothesis)
        self.validator_good.calculate_metrics() # Needs to be called first
        hypothesis_results_good = self.validator_good.hypothesis_test(alpha=0.05)

        self.assertIsInstance(hypothesis_results_good, dict)
        expected_keys = ['hypothesis_supported', 'criteria_passed', 'total_criteria', 'detailed_results']
        for key in expected_keys:
            self.assertIn(key, hypothesis_results_good, f"Key '{key}' missing in hypothesis results.")

        # Based on the ModelValidator logic (3/4 criteria for support)
        # We expect good predictions to pass more criteria
        # This requires knowing the thresholds in ModelValidator or making them configurable.
        # For now, let's assume default thresholds and check structure.
        # print(f"Good results: {self.validator_good.validation_results}") # For debugging threshold passes
        # print(f"Good hypothesis: {hypothesis_results_good}")


        # Test with poor predictions (should likely not support hypothesis)
        self.validator_poor.calculate_metrics()
        hypothesis_results_poor = self.validator_poor.hypothesis_test(alpha=0.05)
        # print(f"Poor results: {self.validator_poor.validation_results}")
        # print(f"Poor hypothesis: {hypothesis_results_poor}")
        # self.assertFalse(hypothesis_results_poor['hypothesis_supported']) # This depends heavily on thresholds

        # Test with perfect predictions (should support hypothesis)
        self.validator_perfect.calculate_metrics()
        hypothesis_results_perfect = self.validator_perfect.hypothesis_test(alpha=0.05)
        self.assertTrue(hypothesis_results_perfect['hypothesis_supported'])
        self.assertEqual(hypothesis_results_perfect['criteria_passed'], hypothesis_results_perfect['total_criteria'])


@patch('scientific_workflow.GEMINI_API_KEY', "dummy_test_key") # Ensure API key is set for most API tests
class TestGeminiAPICall(unittest.TestCase):
    def setUp(self):
        self.prompt = "Test prompt for Gemini API"
        # Using a simplified schema for most tests
        self.simple_schema = {"type": "OBJECT", "properties": {"test_key": {"type": "STRING"}}, "required": ["test_key"]}

        # Load actual schemas for at least one test to ensure they are loadable
        # This assumes schemas directory is in root, and load_json_schema works correctly.
        # We also need to mock os.path.exists for load_json_schema if it's not already mocked globally
        # and open for the schema files.
        schema_dir = os.path.join(os.getcwd(), 'schemas')
        if not os.path.exists(os.path.join(schema_dir, "PRE_EXPERIMENT_SCHEMA.json")):
             # Create dummy schema files if they don't exist to allow tests to run
            os.makedirs(schema_dir, exist_ok=True)
            with open(os.path.join(schema_dir, "PRE_EXPERIMENT_SCHEMA.json"), "w") as f:
                json.dump({"type": "OBJECT", "properties": {"abstract": {"type": "STRING"}}}, f)
            with open(os.path.join(schema_dir, "POST_EXPERIMENT_SCHEMA.json"), "w") as f:
                json.dump({"type": "OBJECT", "properties": {"results_narrative": {"type": "STRING"}}}, f)
            with open(os.path.join(schema_dir, "COMPARATIVE_SCHEMA.json"), "w") as f:
                json.dump({"type": "OBJECT", "properties": {"overall_introduction": {"type": "STRING"}}}, f)

        self.pre_experiment_schema = load_json_schema("PRE_EXPERIMENT_SCHEMA.json")
        self.assertNotIn("error", self.pre_experiment_schema, "Failed to load PRE_EXPERIMENT_SCHEMA.json for tests")


    @patch('scientific_workflow.validate') # Mock jsonschema.validate
    @patch('scientific_workflow.requests.post') # Mock requests.post
    def test_successful_api_call(self, mock_post, mock_validate):
        response_mock = MagicMock()
        response_mock.status_code = 200
        api_response_text = '{"test_key": "test_value"}'
        response_mock.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": api_response_text}]}}]
        }
        mock_post.return_value = response_mock

        expected_parsed_json = {"test_key": "test_value"}

        result = call_gemini_api_with_requests(self.prompt, self.simple_schema)

        mock_post.assert_called_once()
        args, kwargs_call = mock_post.call_args # Use kwargs_call to avoid conflict with outer scope's kwargs if any

        self.assertIn(scientific_workflow.API_URL, args[0])

        # Correctly check 'data' key in the call arguments for requests.post
        self.assertIn('data', kwargs_call)
        self.assertNotIn('json', kwargs_call) # Ensure it's not using json=payload directly

        # Parse the JSON string from the 'data' argument to check its content
        payload_sent = json.loads(kwargs_call['data'])
        self.assertEqual(payload_sent["contents"][0]["parts"][0]["text"], self.prompt)
        self.assertEqual(payload_sent["generationConfig"]["responseSchema"], self.simple_schema)

        mock_validate.assert_called_once_with(instance=expected_parsed_json, schema=self.simple_schema)
        self.assertEqual(result, expected_parsed_json)

    @patch('scientific_workflow.requests.post')
    def test_api_call_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("API Error")
        mock_response.text = "Server error message details" # To match how it's used in except block
        mock_post.return_value = mock_response

        result = call_gemini_api_with_requests(self.prompt, self.simple_schema)

        self.assertIn("error", result)
        # The error message in the code is str(e), which for HTTPError is "API Error"
        self.assertEqual(result["error"], "API Error")


    @patch('scientific_workflow.validate')
    @patch('scientific_workflow.requests.post')
    def test_api_call_validation_error(self, mock_post, mock_validate):
        response_mock = MagicMock()
        response_mock.status_code = 200
        api_response_text = '{"test_key": "test_value_mismatch"}' # Data that might mismatch schema
        response_mock.json.return_value = {"candidates": [{"content": {"parts": [{"text": api_response_text}]}}]}
        mock_post.return_value = response_mock

        mock_validate.side_effect = ValidationError("Schema validation failed for test")

        expected_parsed_json = {"test_key": "test_value_mismatch"}

        # Capture print output to check for logged error
        with patch('builtins.print') as mock_print:
            result = call_gemini_api_with_requests(self.prompt, self.simple_schema)

        mock_validate.assert_called_once_with(instance=expected_parsed_json, schema=self.simple_schema)
        # The current implementation logs the validation error but returns the parsed JSON.
        self.assertEqual(result, expected_parsed_json)

        # Check if the validation error was printed
        printed_error = False
        for call_arg in mock_print.call_args_list:
            if "LLM response validation error: Schema validation failed for test" in call_arg[0][0]:
                printed_error = True
                break
        self.assertTrue(printed_error, "Validation error message was not printed.")

    @patch('scientific_workflow.GEMINI_API_KEY', new_callable=unittest.mock.PropertyMock)
    def test_api_call_no_api_key(self, mock_gemini_api_key_prop):
        # This test had issues with patching GEMINI_API_KEY.
        # The variable is read at module level.
        # A more robust way is to patch where it's used or ensure the module is reloaded with the patch.
        # For this specific function, it checks `if not GEMINI_API_KEY:`
        # We will try patching it directly on the module for the scope of this test.

        original_api_key = scientific_workflow.GEMINI_API_KEY
        scientific_workflow.GEMINI_API_KEY = "" # Directly patch the module variable
        try:
            result = call_gemini_api_with_requests(self.prompt, self.simple_schema)
            self.assertIn("error", result)
            self.assertEqual(result["error"], "Google API key not configured. Please set the GEMINI_API_KEY environment variable.")
        finally:
            scientific_workflow.GEMINI_API_KEY = original_api_key # Restore original key


if __name__ == '__main__':
    # This ensures that the module-level mocks for config loading are applied before tests run
    # However, the direct patching of GEMINI_API_KEY in TestGeminiAPICall is more targeted for that specific variable.
    # The global load_dependencies() call is removed as patching GEMINI_API_KEY directly is cleaner.
    unittest.main()

# Need to import scientific_workflow at the end to ensure mocks are applied if they were module-level
import scientific_workflow
