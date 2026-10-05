# Phase 1 Final Review Checklist

## Repository
- [ ] Repo named exactly `SW485-Project-Group#` (with your real group number)
- [ ] Structure matches the handbook (README, 2 notebooks, Dataset/, Supervised_Learning/, docs/)
- [ ] Link accessible to the instructor (public or access granted)
- [ ] No API keys, secrets, or junk files committed

## README
- [ ] All member names and IDs filled in
- [ ] Title and motivation
- [ ] AI-tool usage statement completed

## Dataset folder
- [ ] `raw_kaggle_data.csv` (renamed from aug_train.csv)
- [ ] `preprocessed_data.csv`
- [ ] `dataset_info.txt` has URL, license, original purpose

## Phase1_Data_Exploration.ipynb
- [ ] Introduction & Problem Statement
- [ ] Dataset Selection & Justification
- [ ] Initial Data Inspection
- [ ] EDA with visualizations (class distribution, missing values, distributions, correlations, feature vs target)
- [ ] Preprocessing Pipeline with justifications
- [ ] Preprocessed Dataset Ready for Modeling
- [ ] Key Insights & Challenges

## Phase1_Supervised_Learning.ipynb
- [ ] Model Selection Rationale (at least 2 algorithms)
- [ ] Well-commented implementation & training
- [ ] Hyperparameter tuning process & results
- [ ] Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
- [ ] 5- or 10-fold cross-validation
- [ ] Confusion matrix heatmap, ROC curves, feature importance (tree models)
- [ ] Model comparison, misclassification analysis, trade-offs
- [ ] Conclusion: best model & reasoning
- [ ] Next steps for Phase 2

## Quality checks
- [ ] Notebooks run top-to-bottom (Restart & Run All)
- [ ] Outputs/plots visible in saved notebooks
- [ ] Class imbalance handled; evaluation not based on accuracy alone
- [ ] No data leakage (split before scaling/encoding/resampling; `enrollee_id` excluded)
- [ ] Task distribution documented; one member submits the link on LMS before Oct 22, 2026
