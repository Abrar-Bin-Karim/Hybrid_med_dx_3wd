# Hybrid Intelligent System for Medical Diagnosis Support

This repository contains the implementation of a medical decision-support system
based on Three-Way Decisions (3WD) and Explainable AI (SHAP).

## Project Overview
- Dataset: UCI Wisconsin Breast Cancer (Diagnostic)
- Baseline Model: Logistic Regression
- Decision Framework: Three-Way Decisions (Confirm / Rule out / Uncertain)
- Explainability: SHAP (in progress)

## Folder Structure
- data/raw/        : Original and cleaned datasets
- notebooks/       : Experiments and analysis
- src/             : Core implementation
- results/         : Saved predictions and evaluation results
- reports/figures/ : Thesis-ready plots

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
