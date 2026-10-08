# Task Distribution

## Phase 1 (Deadline: Oct 22, 2026)

| Member | Responsibility | Deliverable | Internal Deadline |
|--------|----------------|-------------|-------------------|
| Manar Alrazin | Problem statement, README, repo setup, dataset justification, notebook intro, merging notebooks & final review | README.md, repo, dataset_info.txt, notebook sections 1-2, final QA | Oct 8 (setup), Oct 19-21 (review) |
| Noura Alaqeel | Data inspection, statistics, missing values handling, key insights | Notebook: Initial Data Inspection, Key Insights & Challenges | Oct 8 |
| Rana Albarrak | EDA visualizations + interpretation | Notebook: Exploratory Data Analysis | Oct 12 |
| Tojan Alnajjar | Encoding, scaling, outliers, feature engineering, train/test split, imbalance handling | Notebook: Preprocessing Pipeline, preprocessed_data.csv | Oct 12 |
| Munira Algaith | Model 1: rationale, training, tuning, feature importance, error analysis | Phase1_Supervised_Learning.ipynb (Model 1 sections) | Oct 15 |
| Ghala Alsanee | Model 2: rationale, training, tuning, error analysis | Phase1_Supervised_Learning.ipynb (Model 2 sections) | Oct 15 |
| Fatimah Omran | Evaluation, comparison, conclusion, next steps for Phase 2 | Metrics, ROC curves, confusion matrices, comparison & conclusion | Oct 18 |


## Shared rules
- Same train/test split, `random_state=42`, and stratified 5-fold CV for both models.
- Drop `enrollee_id`; split before scaling, encoding, or resampling.
- Report precision, recall, F1, confusion matrix and ROC-AUC (not accuracy alone).
- Model owners save predictions or fitted models for the comparison.

## Phase 2 (Deadline: Dec 3, 2026)
_To be assigned after Phase 1 submission. Roles will rotate._

| Member | Responsibility | Deliverable | Internal Deadline |
|--------|----------------|-------------|-------------------|
| | Unsupervised learning (clustering) | Phase2_Unsupervised_Learning.ipynb | |
| | Generative AI & prompt templates | Phase2_Generative_AI.ipynb, prompt_design_rationale.md | |
| | Executive dashboard | Dashboard link in README | |

## Individual Reflections
_Each member adds a short reflection on what they learned and the challenges faced._

### Manar Alrazin
### Noura Alaqeel
My part was data inspection, statistics, missing values handling and key insights.
The most important thing I learned is that missing values are not always just "empty" data. In our dataset, candidates who did not report their company were about twice as likely to look for a new job (38.8% vs 18.4%), so instead of deleting these rows or filling them with the most common value, I kept them as an "Unknown" category.
I also learned what data leakage means and why we should not use mean or mode imputation before the train/test split.
It was also my first time working with notebooks in VS Code, so setting up the environment took me some time.
The hardest part for me was deciding how to handle each column with missing values: when to fill it with "Unknown", when to use "No Major" and when to drop the rows.
### Rana Albarrak
### Tojan Alnajjar
### Munira Algaith
### Ghala Alsanee
### Fatimah Omran
