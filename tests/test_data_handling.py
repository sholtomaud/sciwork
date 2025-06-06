import unittest
import json
import os
import numpy as np
import pandas as pd

class TestDataHandling(unittest.TestCase):

    def setUp(self):
        """Set up paths for the tests."""
        # Assuming tests are run from the repository root directory
        self.test_data_dir = os.path.join(os.getcwd(), 'test_data')
        self.scenarios = ['support', 'fail', 'marginal']
        self.expected_data_files = [f"{s}_data.json" for s in self.scenarios]
        self.expected_top_level_keys = ['flow_inputs', 'time_steps', 'observed_data']
        self.expected_observed_data_keys = [
            'storage_level', 'overflow', 'energy_input',
            'storage_loss', 'time', 'power_output', 'energy_not_served' # Added all keys from generate_test_data
        ]

    def test_data_files_exist(self):
        """Test if all expected data files exist in the test_data directory."""
        self.assertTrue(os.path.isdir(self.test_data_dir),
                        f"Test data directory not found at: {self.test_data_dir}")
        for file_name in self.expected_data_files:
            file_path = os.path.join(self.test_data_dir, file_name)
            self.assertTrue(os.path.exists(file_path), f"Data file not found: {file_path}")

    def test_data_loading_and_structure(self):
        """Test loading of data files and their basic structure."""
        for scenario in self.scenarios:
            file_name = f"{scenario}_data.json"
            file_path = os.path.join(self.test_data_dir, file_name)

            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    self.fail(f"Failed to decode JSON from {file_path}: {e}")

            self.assertIsInstance(data, dict, f"Data in {file_name} is not a dictionary.")

            for key in self.expected_top_level_keys:
                self.assertIn(key, data, f"Key '{key}' missing in {file_name}.")

            observed_data = data['observed_data']
            self.assertIsInstance(observed_data, dict, f"'observed_data' in {file_name} is not a dictionary.")

            for key in self.expected_observed_data_keys:
                self.assertIn(key, observed_data, f"Key '{key}' missing in 'observed_data' in {file_name}.")

            # Check that main data arrays are lists and not empty
            self.assertIsInstance(data['flow_inputs'], list, f"'flow_inputs' in {file_name} is not a list.")
            self.assertTrue(len(data['flow_inputs']) > 0, f"'flow_inputs' in {file_name} is empty.")

            self.assertIsInstance(data['time_steps'], list, f"'time_steps' in {file_name} is not a list.")
            self.assertTrue(len(data['time_steps']) > 0, f"'time_steps' in {file_name} is empty.")

            for obs_key in self.expected_observed_data_keys:
                self.assertIsInstance(observed_data[obs_key], list,
                                      f"'observed_data']['{obs_key}'] in {file_name} is not a list.")
                # It's possible for some observed data like 'overflow' to be all zeros, so allow empty if appropriate
                # For 'storage_level', 'time', 'energy_input' we expect values.
                if obs_key in ['storage_level', 'time', 'energy_input', 'power_output']:
                    self.assertTrue(len(observed_data[obs_key]) > 0,
                                    f"'observed_data']['{obs_key}'] in {file_name} is empty.")


    def test_data_type_conversion(self):
        """Test basic data type conversions to numpy array and pandas Series."""
        # Load data for one scenario (e.g., 'support')
        scenario_to_test = 'support'
        file_name = f"{scenario_to_test}_data.json"
        file_path = os.path.join(self.test_data_dir, file_name)

        if not os.path.exists(file_path):
            self.fail(f"Test data file for type conversion not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Test flow_inputs conversion to numpy array
        flow_inputs_list = data.get('flow_inputs')
        self.assertIsNotNone(flow_inputs_list, "'flow_inputs' missing for type conversion test.")
        self.assertIsInstance(flow_inputs_list, list)

        flow_inputs_np = np.array(flow_inputs_list)
        self.assertIsInstance(flow_inputs_np, np.ndarray,
                              "Conversion of 'flow_inputs' to numpy array failed.")
        self.assertTrue(flow_inputs_np.size > 0, "'flow_inputs' numpy array is empty.")

        # Test observed_data['storage_level'] conversion to pandas Series
        observed_data_dict = data.get('observed_data')
        self.assertIsNotNone(observed_data_dict, "'observed_data' missing for type conversion test.")
        self.assertIsInstance(observed_data_dict, dict)

        storage_level_list = observed_data_dict.get('storage_level')
        self.assertIsNotNone(storage_level_list, "'observed_data[storage_level]' missing for type conversion test.")
        self.assertIsInstance(storage_level_list, list)

        storage_level_series = pd.Series(storage_level_list)
        self.assertIsInstance(storage_level_series, pd.Series,
                               "Conversion of 'observed_data[storage_level]' to pandas Series failed.")
        self.assertTrue(not storage_level_series.empty, "'storage_level' pandas Series is empty.")

if __name__ == '__main__':
    unittest.main()
