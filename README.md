# XAI for Transportation Safety

**Option 2 - XAI Applications**  
**Rana Dubauskas & Anjali Kota**

This project trains an explainable machine learning model on the **US Accidents** dataset and focuses on transportation safety analysis with interpretable machine learning. The original proposal frames the work around transportation agencies using data-driven safety analysis, the need for actionable explanations, and the gap between strong predictive models and planner-centered interpretability.

The proposal emphasizes three key motivations: transportation agencies increasingly rely on data-driven safety analysis, many ML models are difficult to interpret, and safety decisions require more than a prediction because explanations help support trust, action, and stakeholder justification. fileciteturn2file0L10-L18

## Project goal

The current implementation uses the **US Accidents** dataset to build an explainable model for accident severity prediction. This keeps the project aligned with the proposal's broader transportation-safety XAI goals while using a dataset with rich contextual features.

The current task is:

- Predict whether an accident is **high severity**
- Target definition: `high_severity = 1` if `Severity >= 3`, else `0`
- Main model: **XGBoost**
- Explanations: **SHAP** summary plot and feature-importance output

This setup is consistent with the project’s larger XAI framing: using machine learning plus explanation methods to better understand transportation risk and support practical decision-making. The proposal also highlights the importance of distinguishing true safety risk from confounding factors such as traffic exposure, as well as balancing predictive performance with interpretability. fileciteturn2file0L19-L32

## Project background

According to the proposal and presentation, the project is motivated by several challenges in transportation safety:

- separating true safety risk from traffic exposure
- handling correlated variables such as weather, traffic density, time of day, speed, and road type
- balancing model accuracy with interpretability for real users such as planners and safety personnel

The presentation also identifies a gap in prior work: many studies emphasize predictive performance and feature importance, but place less focus on whether explanations are actually useful for real planners or whether they distinguish true hazards from high-volume exposure. fileciteturn2file0L19-L46

## Current modeling setup

The original proposal described a plan to use SHAP, ICE plots, counterfactual explanations, calibration metrics, and a logistic regression baseline. The current codebase implements the core part of that plan with:

- an **XGBoost** classifier
- predictive metrics
- calibration metrics
- **SHAP** explanations

The proposal’s evaluation plan specifically mentions:
- ROC-AUC
- calibration metrics
- logistic regression baseline comparison
- ICE plots
- global/local SHAP
- counterfactual explanations fileciteturn2file0L47-L56

At this stage, the code is focused on the main XGBoost + SHAP pipeline first, with some of the optional extensions left for future improvement.

## Dataset

This project uses the **US Accidents** dataset from Kaggle.

Expected file location:

```text
data/US_Accidents_March23.csv
```

Dataset link:

```text
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
```

## Project structure

```text
project/
├── data/
│   └── US_Accidents_March23.csv
├── artifacts/
├── train_xai_us_accidents.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run

### 1. Create and activate a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create the required folders if they do not already exist

#### macOS / Linux

```bash
mkdir -p data artifacts
```

#### Windows PowerShell

```powershell
mkdir data
mkdir artifacts
```

### 4. Put the dataset file in the `data` folder

Expected path:

```text
data/US_Accidents_March23.csv
```

### 5. Run the training script

```bash
python train_xai_us_accidents.py
```

## What the script does

The script:

- loads the US Accidents dataset
- creates time-based features from `Start_Time`
- creates the binary target `high_severity`
- selects numeric and categorical transportation/environmental features
- trains an XGBoost classifier
- evaluates the model with predictive and calibration metrics
- saves SHAP explanation outputs

This reflects the project’s broader goal of combining predictive modeling with explanation methods that make transportation safety analysis more understandable and actionable. The proposal specifically emphasizes global and local SHAP explanations and asks whether risk is driven by structural or environmental factors versus confounding factors. fileciteturn2file0L35-L46

## Outputs

After running, the script saves results in the `artifacts/` folder:

- `metrics.json`
- `reliability_diagram.png`
- `test_predictions.csv`
- `xgb_pipeline.joblib`
- `shap_summary.png`
- `shap_feature_importance.csv`
- `top_risk_cases.csv`

## Recommended workflow

1. Start with a smaller `SAMPLE_SIZE` in the script if your laptop is slow.
2. Run the script once to confirm the pipeline works.
3. Inspect `metrics.json` and `shap_summary.png`.
4. Use `shap_feature_importance.csv` and `top_risk_cases.csv` in your report/demo.

## Common issues

### 1. File not found error

Make sure the dataset is exactly here:

```text
data/US_Accidents_March23.csv
```

### 2. Package not found

Make sure the virtual environment is activated before running:

```bash
source .venv/bin/activate
```

Then reinstall dependencies:

```bash
pip install -r requirements.txt
```

### 3. Script is too slow or runs out of memory

Open `train_xai_us_accidents.py` and reduce:

```python
SAMPLE_SIZE = 300_000
```

Try values like:

```python
SAMPLE_SIZE = 50000
```

or

```python
SAMPLE_SIZE = 100000
```

### 4. SHAP step is slow

This is normal on larger samples. The script already limits the SHAP subset, but reducing `SAMPLE_SIZE` can still help.

## Notes for the report

This setup uses the US Accidents dataset for an **accident severity prediction** task rather than accident-vs-no-accident prediction, because the dataset contains accident events and contextual features but does not directly include non-accident observations.

A simple framing for the project is:

> We use the US Accidents dataset to predict whether a reported accident is high severity using transportation, temporal, and environmental features, and apply SHAP to explain the model's predictions.

You can also tie this to the original proposal by noting that the project is still aimed at transportation safety decision support, interpretability, and explanation quality for real users such as planners and safety personnel. The proposal and presentation both emphasize trust, actionability, and the need to distinguish true hazards from misleading high-exposure patterns. fileciteturn2file0L10-L18 fileciteturn2file0L19-L46

## Optional next improvements

Useful extensions if you want a stronger final project:

- add a logistic regression baseline
- add random forest comparison
- filter to junction-related accidents only
- add SHAP waterfall plots for individual cases
- add calibration methods such as isotonic regression or Platt scaling
- add ICE plots
- add counterfactual explanations

These extensions are directly aligned with the original proposal’s evaluation plan and solution vision. fileciteturn2file0L47-L56

## Proposal context

The uploaded proposal presentation describes the project as **“XAI for Transportation Safety”** and frames it around:

- problem and motivation
- challenges
- related work and research gap
- proposed solution using explainable ML
- evaluation with predictive and interpretability metrics
- a staged project timeline

Those themes should stay reflected in the final report and demo even though the implementation now uses the US Accidents dataset instead of the originally proposed UK Road Safety dataset. The presentation title and structure appear on pages 1 through 7. fileciteturn2file0L1-L56
