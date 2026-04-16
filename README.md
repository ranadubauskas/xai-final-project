# XAI for Transportation Safety

**Option 2 - XAI Applications**  
**Rana Dubauskas & Anjali Kota**

This project trains an explainable machine learning model on the **US Accidents** dataset and focuses on transportation safety analysis with interpretable machine learning.

Transportation agencies increasingly rely on data-driven safety analysis, but many machine learning models are difficult to interpret. In safety-critical settings, decision-makers need more than a prediction alone. Explanations can help support trust, actionability, and stakeholder justification.


## Project Goal

This project uses the **US Accidents** dataset to build an explainable model for accident severity prediction using transportation, temporal, and environmental features.


The current task is:

- Predict whether an accident is **high severity**
- Target definition: `high_severity = 1` if `Severity >= 3`, else `0`
- Main model: **XGBoost**
- Explanations: **SHAP** summary plot and feature-importance output

This setup is designed to support transportation safety analysis with interpretable machine learning, while examining how environmental, roadway, and temporal features relate to accident severity.

## Project Background

This project is motivated by several challenges in transportation safety:

- separating true safety risk from traffic exposure
- handling correlated variables such as weather, traffic density, time of day, speed, and road type
- balancing model accuracy with interpretability for real users such as planners and safety personnel

A key gap in prior work is that many studies emphasize predictive performance and feature importance, but place less focus on whether explanations are useful for real transportation decision-making or whether they distinguish true hazards from high-volume exposure.

## Current Modeling Setup

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
- counterfactual explanations

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

## Project Structure

```text
project/
├── data/
│   └── US_Accidents_March23.csv
├── artifacts/
│   ├── metrics.json
│   ├── reliability_diagram.png
│   ├── test_predictions.csv
│   ├── shap_summary.png
│   ├── shap_feature_importance.csv
│   ├── top_risk_cases.csv
│   └── xgb_pipeline.joblib
├── train_xai_us_accidents.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

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

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
1. Go to dataset [link](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
2. Scroll down to `US_Accidents_March23.csv`, select all 46 columns, and click download


### 4. Move the Dowloaded Dataset file in the `data` folder

Expected path:

```text
data/US_Accidents_March23.csv
```

### 5. Run the Training Script

```bash
python train_xai_us_accidents.py
```

## What the Script Does

The script:

- loads the US Accidents dataset
- creates time-based features from `Start_Time`
- creates the binary target `high_severity`
- selects numeric and categorical transportation/environmental features
- trains an XGBoost classifier
- evaluates the model with predictive and calibration metrics
- saves SHAP explanation outputs

This reflects the project’s broader goal of combining predictive modeling with explanation methods that make transportation safety analysis more understandable and actionable. The proposal specifically emphasizes global and local SHAP explanations and asks whether risk is driven by structural or environmental factors versus confounding factors.

## Outputs

After running, the script saves results in the `artifacts/` folder:

- `metrics.json`
- `reliability_diagram.png`
- `test_predictions.csv`
- `xgb_pipeline.joblib`
- `shap_summary.png`
- `shap_feature_importance.csv`
- `top_risk_cases.csv`

### Pre-trained Model

A trained XGBoost pipeline is saved in:

```text
artifacts/xgb_pipeline.joblib
```

This allows users to inspect or reuse the trained model without retraining from scratch.


## Results

*All images and artifacts can be found in the artifacts folder.*

The current XGBoost model shows strong ranking performance and good probability calibration on the accident severity prediction task.

### Quantitative Results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8611 |
| PR-AUC | 0.6220 |
| F1 | 0.5267 |
| Precision | 0.6885 |
| Recall | 0.4265 |
| Brier Score | 0.1078 |
| ECE | 0.0082 |

### Analysis

- The model achieves a **ROC-AUC of 0.8611**, indicating strong ability to distinguish higher-severity from lower-severity accidents.
- The **PR-AUC of 0.6220** is useful because the positive class is not dominant, so precision-recall performance is important.
- The model has **high precision (0.6885)** but more moderate **recall (0.4265)**, meaning it is relatively conservative: when it predicts high severity, it is often correct, but it misses some high-severity cases.
- The **Brier score (0.1078)** and **ECE (0.0082)** indicate that predicted probabilities are well calibrated.

### Reliability Diagram

The reliability diagram shows that predicted probabilities track observed outcome frequencies closely, with the calibration curve staying near the diagonal. This suggests the model’s confidence estimates are reliable.

![Reliability Diagram](artifacts/reliability_diagram.png)

### SHAP Summary Plot

The SHAP summary plot highlights which features most strongly influence the model’s predictions. The most important features include accident distance, longitude, latitude, traffic signal presence, wind chill, weather condition, and crossing-related indicators. The plot below provides a global explanation of model behavior across the evaluation sample. Top global SHAP features include `Distance(mi)`, `Start_Lng`, `Start_Lat`, `Traffic_Signal`, `Wind_Chill(F)`, `Weather_Condition_Fair`, and `Crossing`. :contentReference[oaicite:0]{index=0}

![SHAP Summary Plot](artifacts/shap_summary.png)

### High-Risk Predictions

A few representative high-confidence predictions from the test set include:

| State | Distance (mi) | Weather | Junction | Traffic Signal | y_true | y_prob | y_pred |
|---|---:|---|---:|---:|---:|---:|---:|
| IL | 0.000 | Partly Cloudy | 1 | 0 | 1 | 0.9560 | 1 |
| GA | 0.000 | Overcast | 0 | 0 | 1 | 0.9454 | 1 |
| KY | 5.900 | Clear | 0 | 0 | 1 | 0.9445 | 1 |

These examples show that the model can assign very high predicted probabilities to true high-severity accidents under certain roadway, weather, and temporal conditions.

### Key Takeaways

- The model performs well overall and is especially strong at ranking accident severity risk.
- Calibration is a major strength: predicted probabilities appear trustworthy for decision support.
- Spatial variables and roadway-context indicators are among the strongest drivers of prediction.
- SHAP improves interpretability by showing how environmental, location-based, and roadway-related features influence severity predictions.

### Interpretation

From the SHAP importance output, the strongest contributors to the model’s prediction of whether an accident is high severity are:
1. **Length of Roadway Affected by the Accident:** `Distance (mi)`
2. **Spatial Location:** `Start_Lng` & `Start_Lat`
3. **Roadway Context:** `Traffic_Signal`, `Crossing`, `Stop`, `Junction`
4. **Time-related Features:** `hour_sin`, `hour_cos`, `dayofweek`, `month`
5. **Regional Indicators:** state features such as `CA`, `SC`, and `GA` appear among the top drivers.

The SHAP analysis shows that the model’s severity predictions are influenced by a mix of transportation, environmental, temporal, and spatial factors. The most important global features include incident distance, latitude, longitude, traffic-signal presence, crossing-related indicators, wind chill, weather condition, pressure, temperature, and time-based variables. This indicates that the model is not relying on a single signal, but instead incorporates roadway context, environmental conditions, and timing effects when estimating severity risk. For non-AI users such as transportation planners, these explanations make the model more transparent by showing which factors are associated with elevated risk and how roadway and environmental conditions may contribute to more severe accidents.

## Common Issues

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

## Future Improvements

Useful extensions if you want a stronger final project:

- add a logistic regression baseline
- add random forest comparison
- filter to junction-related accidents only
- add SHAP waterfall plots for individual cases
- add calibration methods such as isotonic regression or Platt scaling
- add ICE plots
- add counterfactual explanations

These extensions are directly aligned with the original proposal’s evaluation plan and solution vision.

