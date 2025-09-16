**Model Training & Coefficient Interpretation Workflow**
An interactive Streamlit app that walks through a 14‑step machine learning workflow — from raw data exploration to model training, coefficient interpretation, stability checks, and final validation. 
Inspired by Scikit‑learn’s coefficient interpretation guide https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py

**Features**
Step‑by‑step workflow (0 → 14) covering:
Data upload & exploration (numerical + categorical features)
Model training & prediction error analysis
Coefficient inspection, scaling, and normalization
Cross‑validation for stability checks
Collinearity diagnostics (AGE vs EXPERIENCE)
Feature‑drop experiments to test robustness
Final performance evaluation (MedAE, R², error plots)
Interactive visualizations for every stage
Transparent & reproducible process for stakeholders and evaluators

**Workflow Phases**
Phase 1 — Data & Exploration (Steps 0–4): Upload data, explore numerical & categorical features, visualize relationships.
Phase 2 — Training & Interpretation (Steps 5–10): Train model, evaluate errors, inspect coefficients, scale & normalize.
Phase 3 — Stability & Validation (Steps 11–14): Cross‑validation, collinearity checks, feature‑drop experiments, final performance.

Key Learnings
Scaling coefficients is essential for fair comparison
Correlated features destabilize models (AGE vs EXPERIENCE)
Cross‑validation reveals coefficient robustness
Ridge regularization improves stability without losing accuracy
Coefficients ≠ causality — interpret with caution

👥 Collaborators
Special thanks to:

Vinit Sutar
Chirag Jhumkawala
Sandhya Hinduja

for their support and collaboration during the session.

