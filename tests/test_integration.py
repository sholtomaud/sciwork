import unittest
import os
import shutil
import json
from unittest.mock import patch, MagicMock

# Adjust import path assuming tests are run from the repository root
import scientific_workflow # Import the module itself to access its (patched) globals
# from scientific_workflow import run_scientific_workflow # Already done by importing scientific_workflow if __all__ is not used.
# No direct import of schema constants from scientific_workflow, will access them via scientific_workflow.THE_VAR

# Mock LLM Responses (ensure these are valid according to the schemas)
# Updated MOCK_COMPARATIVE_RESPONSE_1 and MOCK_COMPARATIVE_RESPONSE_2 for 3 scenarios

MOCK_PRE_EXPERIMENT_RESPONSE = {
    "abstract": "This is a test abstract from integration test.",
    "introduction": "This is a test introduction from integration test.",
    "hypothesis_narrative": "The model is hypothesized to be accurate (integration test).",
    "hypothesis_variables": ["Var X", "Var Y"],
    "methods_narrative": "Standard methods were used for integration testing.",
    "methods_components": ["Integration Component A", "Integration Component B"]
}

MOCK_POST_EXPERIMENT_RESPONSE_SUPPORT = {
    "results_narrative": "Results for the 'support' scenario from integration test.",
    "discussion_narrative": "Discussion for the 'support' scenario from integration test.",
    "conclusion_narrative": "Conclusion for the 'support' scenario from integration test.",
    "structured_assessment": {
        "clarity_of_results": "High",
        "comparison_to_hypothesis": "Supported",
        "limitations_of_study": "None noted for this test.",
        "implications_for_future_work": "Further integration."
    }
}

MOCK_COMPARATIVE_RESPONSE_1 = {
    "overall_introduction": "Comparative introduction (call 1) for integration test.",
    "experiment_summaries": [
        {"scenario_name": "support", "key_finding": "KF support placeholder C1", "metrics": {}},
        {"scenario_name": "fail", "key_finding": "KF fail placeholder C1", "metrics": {}},
        {"scenario_name": "marginal", "key_finding": "KF marginal placeholder C1", "metrics": {}}
    ],
    "overall_conclusion": "Comparative conclusion (call 1) placeholder."
}

MOCK_COMPARATIVE_RESPONSE_2 = {
    "overall_introduction": "Refined comparative introduction (call 2) for integration test.",
    "experiment_summaries": [
        {
            "scenario_name": "support",
            "key_finding": "Detailed KF for support (call 2).",
            "metrics": {"rmse": 0.15, "nse": 0.85, "hypothesis_supported": True, "mae": 0.1, "mape": 1.0}
        },
        {
            "scenario_name": "fail",
            "key_finding": "Detailed KF for fail (call 2).",
            "metrics": {"rmse": 0.55, "nse": 0.25, "hypothesis_supported": False, "mae": 0.5, "mape": 10.0}
        },
        {
            "scenario_name": "marginal",
            "key_finding": "Detailed KF for marginal (call 2).",
            "metrics": {"rmse": 0.35, "nse": 0.55, "hypothesis_supported": True, "mae": 0.3, "mape": 5.0}
        }
    ],
    "comparative_analysis": {
        "performance_overview": "Model performance varied across scenarios (integration test).",
        "cross_scenario_insights": "Cross-scenario insights from integration test."
    },
    "overall_conclusion": "Final comparative conclusion (call 2) from integration test."
}


class TestWorkflowIntegration(unittest.TestCase):
    def setUp(self):
        # The workflow will use its default "output" directory.
        # We clean this up before and after the test.
        self.workflow_output_dir = "output"
        if os.path.exists(self.workflow_output_dir):
            shutil.rmtree(self.workflow_output_dir)
        # No need to os.makedirs here, the workflow should do it.

        self.mock_pre_experiment_response = MOCK_PRE_EXPERIMENT_RESPONSE
        self.mock_post_experiment_response_support = MOCK_POST_EXPERIMENT_RESPONSE_SUPPORT
        # It's good practice to have distinct mocks if behavior should differ
        self.mock_post_experiment_response_fail = {**MOCK_POST_EXPERIMENT_RESPONSE_SUPPORT, "results_narrative": "Results for 'fail' scenario (mocked)"}
        self.mock_post_experiment_response_marginal = {**MOCK_POST_EXPERIMENT_RESPONSE_SUPPORT, "results_narrative": "Results for 'marginal' scenario (mocked)"}

        self.mock_comparative_response_1 = MOCK_COMPARATIVE_RESPONSE_1
        self.mock_comparative_response_2 = MOCK_COMPARATIVE_RESPONSE_2
        # GEMINI_API_KEY is patched via decorator now.

    def tearDown(self):
        # Clean up the directory created by the workflow
        if os.path.exists(self.workflow_output_dir):
            shutil.rmtree(self.workflow_output_dir)

    # Removed patch for 'scientific_workflow.output_base_dir'.
    # The test will use the default output dir "output" managed by setUp/tearDown.
    @patch('scientific_workflow.call_gemini_api_with_requests')
    @patch('scientific_workflow.GEMINI_API_KEY', "MOCK_INTEGRATION_TEST_KEY") # Ensure API key check passes
    def test_run_scientific_workflow_integration(self, mock_call_llm, mock_api_key_dummy): # Corrected order of mock arguments

        # Counter for comparative schema calls
        self.comparative_call_count = 0

        def mock_llm_calls_side_effect(prompt_text, schema_arg, *args, **kwargs):
            # The schema_arg passed to call_gemini_api_with_requests is the actual schema dict
            # Compare with the patched global variables in the scientific_workflow module
            if schema_arg == scientific_workflow.PRE_EXPERIMENT_SCHEMA_FROM_FILE:
                return self.mock_pre_experiment_response
            elif schema_arg == scientific_workflow.POST_EXPERIMENT_SCHEMA_FROM_FILE:
                # Provide a generic response for any post-experiment call
                # In a more detailed test, could vary this based on prompt_text content (scenario name)
                # For now, MOCK_POST_EXPERIMENT_RESPONSE_SUPPORT will be used for all scenarios.
                # To make experiment_summaries in MOCK_COMPARATIVE_RESPONSE_2 align, it expects 'support'.
                # This implies the test data for 'fail' and 'marginal' should also lead to some results
                # that can be summarized, even if the mock LLM response is generic.
                # The scenario name is part of the prompt to the LLM for post-experiment.
                if "'support'" in prompt_text: # A bit fragile, but distinguishes
                    return self.mock_post_experiment_response_support
                elif "'fail'" in prompt_text:
                    return self.mock_post_experiment_response_fail
                elif "'marginal'" in prompt_text:
                    return self.mock_post_experiment_response_marginal
                # Fallback, though specific checks should catch all scenarios
                return self.mock_post_experiment_response_support
            elif schema_arg == scientific_workflow.COMPARATIVE_SCHEMA_FROM_FILE:
                self.comparative_call_count += 1
                if self.comparative_call_count == 1:
                    return self.mock_comparative_response_1
                else:
                    return self.mock_comparative_response_2

            # Fallback for safety, though all calls should be caught above
            print(f"Warning: Unmocked LLM call with prompt: {prompt_text[:100]}... and schema: {str(schema_arg)[:100]}...")
            return {"error": "Unknown schema or prompt for mock LLM call in integration test"}

        mock_call_llm.side_effect = mock_llm_calls_side_effect

        # Run the workflow
        run_scientific_workflow()

        # Assertions
        # Expected number of LLM calls:
        # 1 for pre-experiment
        # 3 for post-experiment (support, fail, marginal)
        # 2 for comparative report
        expected_llm_calls = 1 + 3 + 2
        self.assertEqual(mock_call_llm.call_count, expected_llm_calls, f"Expected {expected_llm_calls} LLM calls, got {mock_call_llm.call_count}")

        # Check for key output files using self.workflow_output_dir
        output_path_common = os.path.join(self.workflow_output_dir, "common_initial_sections.json")
        self.assertTrue(os.path.exists(output_path_common), f"File not found: {output_path_common}")
        with open(output_path_common, 'r') as f:
            common_data = json.load(f)
            self.assertEqual(common_data['abstract'], self.mock_pre_experiment_response['abstract'])

        # Check files for all default scenarios
        default_scenarios = ['support', 'fail', 'marginal']
        for scenario_name in default_scenarios:
            scenario_dir = os.path.join(self.workflow_output_dir, scenario_name)
            self.assertTrue(os.path.isdir(scenario_dir), f"Directory not found for scenario: {scenario_name}")

            output_path_pre = os.path.join(scenario_dir, "pre_experiment_llm_response.json")
            output_path_post = os.path.join(scenario_dir, "post_experiment_llm_response.json")
            output_path_report = os.path.join(scenario_dir, f"report_{scenario_name}.docx")
            output_path_plot = os.path.join(scenario_dir, f"plot_{scenario_name}.png")

            self.assertTrue(os.path.exists(output_path_pre), f"File not found: {output_path_pre}")
            self.assertTrue(os.path.exists(output_path_post), f"File not found: {output_path_post}")

            if scenario_name == 'support':
                 with open(output_path_post, 'r') as f:
                    post_data = json.load(f)
                    self.assertEqual(post_data['results_narrative'], self.mock_post_experiment_response_support['results_narrative'])
            elif scenario_name == 'fail':
                 with open(output_path_post, 'r') as f:
                    post_data = json.load(f)
                    self.assertEqual(post_data['results_narrative'], self.mock_post_experiment_response_fail['results_narrative'])
            elif scenario_name == 'marginal':
                 with open(output_path_post, 'r') as f:
                    post_data = json.load(f)
                    self.assertEqual(post_data['results_narrative'], self.mock_post_experiment_response_marginal['results_narrative'])

            self.assertTrue(os.path.exists(output_path_report), f"File not found: {output_path_report}")
            self.assertTrue(os.path.getsize(output_path_report) > 0, f"{output_path_report} is empty.")
            self.assertTrue(os.path.exists(output_path_plot), f"File not found: {output_path_plot}")
            self.assertTrue(os.path.getsize(output_path_plot) > 0, f"{output_path_plot} is empty.")

        # Comparative report files
        comp_dir = os.path.join(self.workflow_output_dir, "comparative_report")
        output_path_comp_resp1 = os.path.join(comp_dir, "comparative_llm_response_1.json")
        output_path_comp_resp2 = os.path.join(comp_dir, "comparative_llm_response_2.json")
        output_path_comp_report = os.path.join(comp_dir, "comparative_report.docx")

        self.assertTrue(os.path.exists(output_path_comp_resp1), f"File not found: {output_path_comp_resp1}")
        with open(output_path_comp_resp1, 'r') as f:
            comp1_data = json.load(f)
            self.assertEqual(comp1_data['overall_introduction'], self.mock_comparative_response_1['overall_introduction'])

        self.assertTrue(os.path.exists(output_path_comp_resp2), f"File not found: {output_path_comp_resp2}")
        with open(output_path_comp_resp2, 'r') as f:
            comp2_data = json.load(f)
            self.assertEqual(comp2_data['overall_introduction'], self.mock_comparative_response_2['overall_introduction'])

        self.assertTrue(os.path.exists(output_path_comp_report), f"File not found: {output_path_comp_report}")
        self.assertTrue(os.path.getsize(output_path_comp_report) > 0, f"{output_path_comp_report} is empty.")


if __name__ == '__main__':
    unittest.main()
