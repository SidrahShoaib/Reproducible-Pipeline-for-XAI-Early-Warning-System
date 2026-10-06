# Reproducible Pipeline for an Explainable AI-Based Early Warning System for At-Risk Students in Foundation-Level STEM Courses

Code accompanying the dissertation *From Prediction to Actionable Support: An Explainable AI-Based Early Warning System for At-Risk Students in Foundation-Level STEM Courses* (ENGE 997, Master of Engineering, Auckland University of Technology, October 2026).

## What this is

A checkpoint-staged machine learning pipeline that predicts academic risk for students enrolled in two foundation-level STEM courses — MATH400 (Foundation Mathematics) and COMP401 (Foundation Problem Solving) — at four points in the semester: Weeks 3, 6, 8 and 12. The pipeline compares Logistic Regression, Random Forest and XGBoost under forward-semester validation, applies Platt calibration where it improves the Brier score, selects operational thresholds against F2, and generates global and case-level SHAP explanations for the selected model.

## The dataset is not included

The data analysed in this study consists of institutional student records released under ethics approval (AUTEC reference 26/237). It was provided to the research team in de-identified form by the programme administrator acting as data custodian, and it **remains the property of Auckland University of Technology**. It is not redistributed here in any form, and no processed or checkpoint-level dataset is uploaded. Any file paths, connection strings or data-loading routines in the scripts below refer to that restricted dataset and will not execute against the public repository. `data/` is gitignored.

## What the code does

| Stage | Purpose |
| --- | --- |
| `00_audit` | Source-data quality audit; column, row and coverage inventory |
| `01_linkage` | De-identified record linkage at the student–course–semester level |
| `02_checkpoints` | Checkpoint construction at Weeks 3, 6, 8, 12 with strict temporal cutoffs |
| `03_features` | Feature engineering; portfolio-heading standardisation, percentage conversion, attendance and engagement aggregation |
| `04_pipeline` | Preprocessing inside the model pipeline (median imputation, missingness indicators, scaling, encoding) to enforce leakage controls |
| `05_validation` | Forward-chaining temporal validation across 2024 S1 → 2025 S1 |
| `06_calibration` | Platt vs. isotonic comparison; Brier-based retention decision |
| `07_thresholds` | F2-optimised threshold selection on development out-of-fold predictions |
| `08_holdout` | Final evaluation on the disjoint 2025 S2 holdout group |
| `09_shap` | Global mean-absolute-SHAP rankings and local decomposition for prototype cases |

## Key design decisions

Three constraints shape the whole pipeline and are worth stating for anyone reading the code.

**Prediction is checkpoint-based, not semester-based.** Each checkpoint is a separate dataset over a separate feature space, and hyperparameters are tuned separately at each. The Week 3 model is not the Week 12 model with fewer observations — it is a different model.

**No predictor may use information unavailable at the moment of prediction.** The temporal rule is enforced at checkpoint construction and again by fitting all preprocessing objects on training data only. Section 3.10.2 of the dissertation explains what this excluded and what it cost.

**Explanations are a methodological output, not an afterthought.** Per-student attribution is a requirement of the design, which is why SHAP is generated for the selected model at every checkpoint rather than added at the end.

## Reproducibility

Random seeds, package versions and search spaces are recorded with each run. Package versions: Python 3.12.10, pandas 3.0.5, scikit-learn 1.9.0, XGBoost 3.4.1, SHAP 0.52.0.

The scripts reproduce the reported results from the archived inputs, subject to the data-sharing restriction above. Because the dataset cannot be redistributed, reproduction requires equivalent institutional access.

## Scope and caveats

This is a retrospective research pipeline, not a deployed system. It ranks students by predicted risk, states contributing factors, and suggests an intervention category. It does not enrol anyone, contact students, touch grades, or make any determination.

Three limitations bear on the code specifically. The final holdout is a student-level split within a single semester rather than a temporal split, so holdout figures are an upper bound on future-cohort performance; the forward folds are the more realistic evidence. Samples are small (41 and 32 holdout records, with approximately 10 and 18 positive cases), so metric differences between models are within noise. And operational thresholds are properties of a particular model's calibration on a particular dataset, not transferable risk cut-offs — an institution adopting this design should set its own operating point against its own review capacity.

## Licence and contact

Code released for academic and research use. No dataset is included or implied. Questions about the method should be directed to the author; questions about the underlying data should be directed to Auckland University of Technology.
