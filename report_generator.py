import os
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import matplotlib.pyplot as plt

def add_heading(doc, text, level, size=None):
    """Adds a styled heading to the document."""
    heading = doc.add_heading(level=level)
    run = heading.add_run(text)
    run.font.name = 'Calibri'
    if size:
        run.font.size = Pt(size)
    else:
        run.font.size = Pt(14) if level == 1 else Pt(12)
    run.bold = True

def add_paragraph(doc, text, style=None):
    """Adds a styled paragraph to the document."""
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(11)

def create_scientific_report(results_summary, initial_report_sections, output_dir='.'):
    """
    Generates a complete scientific report in DOCX format from a hybrid schema.
    """
    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(11)

    # Title
    title = doc.add_heading(level=0)
    title_run = title.add_run('Validation of an Odum Energy Systems Model')
    title_run.bold = True
    title_run.font.size = Pt(18)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Abstract
    add_heading(doc, 'Abstract', level=1)
    # Use .get() with a default empty string for safety
    add_paragraph(doc, initial_report_sections.get('abstract', ''))

    # Introduction
    add_heading(doc, 'Introduction', level=1)
    add_paragraph(doc, initial_report_sections.get('introduction', ''))

    # --- UPDATED HYPOTHESIS SECTION ---
    add_heading(doc, 'Hypothesis', level=1)
    # Safely get the hypothesis object, defaulting to an empty dict
    hypothesis_data = initial_report_sections.get('hypothesis', {})
    
    add_heading(doc, 'Hypothesis Statement', level=2, size=12)
    add_paragraph(doc, hypothesis_data.get('narrative_statement', ''))

    # Handle structured variables
    iv_data = hypothesis_data.get('independent_variable', {})
    dv_data = hypothesis_data.get('dependent_variable', {})

    add_heading(doc, 'Variables', level=2, size=12)
    add_paragraph(doc, 'Independent Variable:', style='List Bullet')
    add_paragraph(doc, f"Name: {iv_data.get('name', 'N/A')}", style='List Bullet 2')
    add_paragraph(doc, f"Description: {iv_data.get('description', 'N/A')}", style='List Bullet 2')
    add_paragraph(doc, f"Unit: {iv_data.get('unit', 'N/A')}", style='List Bullet 2')

    add_paragraph(doc, 'Dependent Variable:', style='List Bullet')
    add_paragraph(doc, f"Name: {dv_data.get('name', 'N/A')}", style='List Bullet 2')
    add_paragraph(doc, f"Description: {dv_data.get('description', 'N/A')}", style='List Bullet 2')
    add_paragraph(doc, f"Unit: {dv_data.get('unit', 'N/A')}", style='List Bullet 2')


    # --- UPDATED METHODS SECTION ---
    add_heading(doc, 'Methods', level=1)
    methods_data = initial_report_sections.get('methods', {})
    
    # Add the main narrative text
    add_paragraph(doc, methods_data.get('narrative', ''))

    # Add the new table of key components
    key_components = methods_data.get('key_components', [])
    if key_components:
        add_heading(doc, 'Summary of Methodological Components', level=2, size=12)
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Component'
        hdr_cells[1].text = 'Purpose in this Study'
        for component in key_components:
            row_cells = table.add_row().cells
            row_cells[0].text = component.get('component_name', 'N/A')
            row_cells[1].text = component.get('purpose_in_study', 'N/A')


    # --- SCENARIO RESULTS (Unchanged) ---
    for scenario, data in results_summary.items():
        doc.add_page_break()
        add_heading(doc, f'Scenario Analysis: {scenario.title()}', level=1)

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle(f'Model Validation Results - {scenario.title()}', fontsize=14)
        
        time_steps = data.get('time_steps', [])
        observations = data.get('observations', [])
        predictions = data.get('predictions', [])

        axes[0,0].plot(time_steps, observations, 'b-', label='Observed', alpha=0.7)
        axes[0,0].plot(time_steps, predictions, 'r--', label='Predicted', alpha=0.7)
        axes[0,0].set_title('Time Series Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].scatter(observations, predictions, alpha=0.6)
        min_val = min(min(observations), min(predictions))
        max_val = max(max(observations), max(predictions))
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
        axes[0,1].set_title('Predicted vs Observed')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        residuals = np.array(observations) - np.array(predictions)
        axes[1,0].scatter(predictions, residuals, alpha=0.6)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_title('Residual Plot')
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Residual Distribution')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(output_dir, f'plot_{scenario}.png')
        plt.savefig(plot_path)
        plt.close(fig)
        doc.add_picture(plot_path, width=Inches(6))

        # Results
        add_heading(doc, 'Results', level=2)
        results_interpretation = data.get('results_interpretation', {})
        results_section = results_interpretation.get('results', {})
        add_paragraph(doc, results_section.get('summary', ''))
        
        # Add stats table
        validation_metrics = data.get('validation_metrics', {})
        hypothesis_results = data.get('hypothesis_results', {})
        stats_data = {
            "Metric": ["RMSE", "MAE", "MAPE (%)", "NSE", "Pearson's r", "p-value (Pearson)", "Hypothesis Supported"],
            "Value": [
                f"{validation_metrics.get('rmse', 0):.3f}",
                f"{validation_metrics.get('mae', 0):.3f}",
                f"{validation_metrics.get('mape', 0):.2f}",
                f"{validation_metrics.get('nse', 0):.3f}",
                f"{validation_metrics.get('pearson_r', 0):.3f}",
                f"{validation_metrics.get('pearson_p', 0):.4f}",
                str(hypothesis_results.get('hypothesis_supported', 'N/A'))
            ]
        }
        df = pd.DataFrame(stats_data)
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Metric'
        hdr_cells[1].text = 'Value'
        for _, row in df.iterrows():
            row_cells = table.add_row().cells
            row_cells[0].text = row['Metric']
            row_cells[1].text = row['Value']

        add_heading(doc, 'Discussion', level=2)
        add_paragraph(doc, results_interpretation.get('discussion', ''))

        add_heading(doc, 'Conclusion', level=2)
        add_paragraph(doc, results_interpretation.get('conclusion', ''))

    # Save the report
    report_path = os.path.join(output_dir, 'scientific_validation_report.docx')
    doc.save(report_path)
    print(f"\nScientific report successfully saved to: {report_path}")
    return report_path