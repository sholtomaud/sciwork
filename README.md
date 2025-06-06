# Automated Scientific Workflow with LLM-Powered Reporting

## Description of the System

This project automates the process of scientific model validation and report generation. It leverages Large Language Models (LLMs) via the Google Gemini API to generate narrative sections of scientific reports, interpret experimental results, and create comparative analyses across multiple scenarios. The system uses JSON schemas to define the expected structure for LLM outputs, ensuring consistency and facilitating data extraction.

The workflow simulates an energy systems model under different conditions, validates its performance, and then uses LLMs to:
1.  Generate initial sections of a scientific report (Abstract, Introduction, Hypothesis, Methods).
2.  Interpret the results of individual experimental scenarios.
3.  Generate a comparative analysis report evaluating the model's performance across all scenarios.

All generated reports are provided in DOCX format, and intermediate LLM responses are saved as JSON files.

## Features

*   **Automated Individual Experiment Reports:** Generates a DOCX report for each experimental scenario, including plots and LLM-generated interpretations.
*   **Automated Comparative Meta-Report:** Generates a single DOCX report that compares and contrasts findings from all scenarios.
*   **Structured LLM Interaction:** Uses JSON schemas to guide LLM output for predictable and parseable results.
*   **JSON Outputs:** Saves all raw LLM responses as JSON files for transparency and reusability.
*   **Schema Validation:** Validates LLM responses against the defined schemas to ensure structural integrity.
*   **Organized Output Directory:** All generated files (JSONs, DOCXs, plots) are stored in a well-defined directory structure.
*   **Modular Design:** Separates concerns for workflow orchestration (`scientific_workflow.py`) and report document generation (`report_generator.py`).

## Directory Structure

```
.
├── schemas/
│   ├── PRE_EXPERIMENT_SCHEMA.json       # Schema for initial LLM call (abstract, intro, etc.)
│   ├── POST_EXPERIMENT_SCHEMA.json      # Schema for interpreting individual scenario results
│   └── COMPARATIVE_SCHEMA.json          # Schema for the comparative analysis report
├── output/
│   ├── common_initial_sections.json     # LLM response for pre-experiment sections
│   ├── support/                         # Example scenario directory
│   │   ├── pre_experiment_llm_response.json  # Copy of common_initial_sections.json
│   │   ├── post_experiment_llm_response.json # LLM interpretation for this scenario
│   │   ├── report_support.docx          # DOCX report for this scenario
│   │   └── plot_support.png             # Plot for this scenario
│   ├── fail/                            # Directory for 'fail' scenario
│   │   └── ...
│   ├── marginal/                        # Directory for 'marginal' scenario
│   │   └── ...
│   └── comparative_report/
│       ├── comparative_llm_response_1.json # First LLM call for comparative report
│       ├── comparative_llm_response_2.json # Second LLM call for comparative report
│       └── comparative_report.docx         # Final comparative DOCX report
├── scientific_workflow.py               # Main script to run the workflow
├── report_generator.py                  # Handles DOCX creation for individual and comparative reports
├── experiment_goals.md                  # User-defined input prompt for the initial LLM call
└── config.ini                           # Configuration file (API keys, model names)
```

## Prerequisites & Setup

### 1. Python
Ensure you have Python 3.x installed (Python 3.7+ recommended).

### 2. Configuration File (`config.ini`)
You need to create a `config.ini` file in the root directory of the project. You can copy `config.ini.template` if it exists, or create it manually. It should contain your Google Gemini API key.

Example `config.ini`:
```ini
[google_ai]
api_key = YOUR_GEMINI_API_KEY_HERE
model_name = gemini-1.5-pro-latest # Or your preferred model
content_api = generateContent
```
Replace `YOUR_GEMINI_API_KEY_HERE` with your actual API key.

### 3. Dependencies
It is highly recommended to use a Python virtual environment to manage dependencies.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
# Activate the virtual environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

Install the required libraries:
```bash
pip install requests python-docx jsonschema numpy pandas matplotlib scipy scikit-learn
```
(Ideally, this project would include a `requirements.txt` file. You can generate one using `pip freeze > requirements.txt` after installing the libraries.)

## How to Run

Once the prerequisites are met and `config.ini` is set up:
```bash
python scientific_workflow.py
```
The script will execute the entire workflow, generate files in the `output/` directory, and print progress messages to the console.

## How it Works (Brief Workflow)

1.  **Initialization:**
    *   Loads configuration from `config.ini`.
    *   Loads JSON schemas from the `schemas/` directory.
2.  **Initial Report Sections (Pre-Experiment LLM Call):**
    *   Reads the user-defined research goals and methodology from `experiment_goals.md`.
    *   Makes the first LLM call (Gemini API) guided by `PRE_EXPERIMENT_SCHEMA.json` to generate:
        *   Abstract
        *   Introduction
        *   Hypothesis (narrative statement, variables)
        *   Methods (narrative, key components)
    *   Saves this response to `output/common_initial_sections.json`.
3.  **Individual Scenario Processing (Loop):**
    *   The workflow iterates through predefined scenarios (e.g., 'support', 'fail', 'marginal').
    *   For each scenario:
        *   **Data Generation & Modeling:** Generates synthetic data, calibrates a system model, simulates its behavior, and performs validation (calculates metrics like RMSE, NSE, etc.).
        *   **Scenario Interpretation (Post-Experiment LLM Call):** Makes a second LLM call guided by `POST_EXPERIMENT_SCHEMA.json`, providing the scenario's statistical data. The LLM generates:
            *   Results Narrative
            *   Discussion Narrative
            *   Conclusion Narrative
        *   **Output Saving:**
            *   Saves the scenario-specific LLM interpretation to `output/<scenario_name>/post_experiment_llm_response.json`.
            *   Copies `common_initial_sections.json` to `output/<scenario_name>/pre_experiment_llm_response.json`.
            *   Generates and saves a visual plot (`plot_<scenario_name>.png`).
            *   Generates and saves an individual DOCX report (`report_<scenario_name>.docx`) combining pre-experiment sections, scenario interpretation, and plots.
4.  **Comparative Report Generation (Post-All Scenarios):**
    *   After all individual scenarios are processed:
        *   **LLM Call 3 (Comparative - Overview):** Makes an LLM call guided by `COMPARATIVE_SCHEMA.json`. It uses the initial introduction, `experiment_goals.md`, and a list of scenarios to draft an `overall_introduction`, a structural `experiment_summaries` section, and a draft `overall_conclusion` for the comparative report. Response saved to `output/comparative_report/comparative_llm_response_1.json`.
        *   **LLM Call 4 (Comparative - Detailed Analysis):** Makes another LLM call, also guided by `COMPARATIVE_SCHEMA.json`. It uses the `overall_introduction` from the previous call and detailed data (metrics, key findings) from all scenarios to:
            *   Fill in the `experiment_summaries` with detailed findings and metrics.
            *   Generate the `comparative_analysis` (performance overview, cross-scenario insights).
            *   Refine the `overall_conclusion`.
        *   Response saved to `output/comparative_report/comparative_llm_response_2.json`.
        *   **DOCX Generation:** Generates the final comparative DOCX report (`comparative_report.docx`) using the content from the second comparative LLM call.

## Proposed Enhancements

**DONE?**
[]   **`requirements.txt`:** Add a `requirements.txt` file for easier dependency management.
[]   **Robust Error Handling:** Implement more comprehensive error handling, logging, and retry mechanisms, especially for API calls.
[]   **Unit and Integration Testing:** Implement unit tests for individual functions and integration tests for the overall workflow. This could include jsonschema validation for LLM responses.
[]   **Configurable Model and Data Input:** Allow users to specify the model to be validated and the source/format of observational data via configuration files, rather than hardcoding them. This includes an option to use pre-generated, immutable synthetic test data for CI/CD and testing.
[]   **Test Data Generation Step:** If configurable observational data is supported, include an optional step to generate synthetic observation data based on user specifications.
[]   **Consolidated JSON Data:** Combine all relevant data (LLM responses, model configuration/parameters, observational data) into a single JSON file per experiment for better data management and reproducibility.
[]   **CI/CD integration.**
[]   **Code linting and formatting.**
[]   **Caching LLM Responses:** Implement a caching mechanism to avoid redundant API calls for identical prompts, saving costs and time.
[]   **Support for different LLM providers.**
[]   **Advanced Comparative Statistics:** Enable the LLM to suggest and interpret appropriate statistical tests for comparing model performance across scenarios (e.g., ANOVA, t-tests on metrics).
[]   **Utilize Structured Assessment:** More directly incorporate the `structured_assessment` part of `POST_EXPERIMENT_SCHEMA.json` into the comparative analysis, perhaps by having the LLM synthesize these structured points.
[]   **Template-based report generation.**
[]   **Version control for experimental results.**
[]   **Interactive data visualization.**
[]   **Interactive Interface:** Develop a simple GUI (e.g., using Tkinter, Streamlit) or a web interface (e.g., Flask, Django) for easier configuration, execution, and viewing of reports.
[]   **Plot Customization:** Allow more control over plot generation through configuration.
[]   **Automated hyperparameter tuning.**
[]   **Sensitivity analysis.**
[]   **Cloud storage integration.**
[]   **Extensible plugin architecture.**
[]   **Scenario-Specific Pre-Experiment Calls:** Option to have the initial LLM call (for abstract, intro, etc.) be specific to each scenario if the overarching goals differ significantly.
[]   **LLM-Powered Peer Review:** Add a workflow step where an LLM acts as a peer reviewer, critiquing the generated report. This would involve providing the reviewer LLM with the report content and relevant statistical data.
[]   **Author Response to Peer Review:** Add a subsequent workflow step for the original LLM "author" to respond to the peer reviewer's critique and revise the report.