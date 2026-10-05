# SW485-Project-Group2 — Executive Decision-Support System for Data Scientist Job-Change Prediction

**Course:** SWE485 – Selected Topics in Software Engineering, King Saud University (1st semester 1448)
**Instructor:** _<instructor name>_

## 1. Team Members

| # | Name | Student ID | Phase 1 Role |
|---|------|-----------|--------------|
| 1 | Manar alrazin | 444200065 | Problem statement, README, repo setup & final review |
| 2 | _<name>_ | _<ID>_ | _<e.g., Dataset selection & EDA>_ |
| 3 | _<name>_ | _<ID>_ | _<e.g., Preprocessing & feature engineering>_ |
| 4 | _<name>_ | _<ID>_ | _<e.g., Supervised model 1>_ |
| 5 | _<name>_ | _<ID>_ | _<e.g., Supervised model 2>_ |
| 6 | _<name>_ | _<ID>_ | _<e.g., Evaluation & comparison>_ |

Detailed task distribution: [`docs/task_distribution.md`](docs/task_distribution.md)

## 2. Project Title and Motivation

**Title:** Predicting Data Scientist Job Change: An Executive Decision-Support System for HR Retention Planning

**Motivation.** Companies that invest in training data scientists need to know which candidates are likely to stay and which are likely to leave for a new job. Losing trained employees raises hiring and training costs and weakens workforce planning. This project develops an executive decision-support system that helps HR leaders predict whether a candidate is looking for a job change, based on their demographics, education, experience, and training history.

Early identification of at-risk candidates helps HR managers:
- reduce recruitment costs,
- improve training investment decisions, and
- design targeted retention strategies.

**Why this dataset?** It is a real-world, structured (tabular) HR dataset with 19,158 records and 13 predictive features (mixed categorical and numeric), a clear binary target, realistic missing values, and class imbalance. This gives us meaningful preprocessing challenges, suits supervised classification (Phase 1), and supports clustering of candidate profiles (Phase 2).

## 3. Problem Statement

HR leaders in organizations that train data scientists lack an early, data-driven signal of which candidates are at risk of leaving. This project builds a binary classifier that predicts whether a candidate is **looking for a job change (1)** or **not (0)**, and presents the result as decision support for HR executives: who to prioritize for retention, where to focus training budgets, and how to plan workforce capacity.

## 4. Dataset

| Item | Detail |
|------|--------|
| Source | [Kaggle – HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists) |
| File used | `aug_train.csv` (stored as `Dataset/raw_kaggle_data.csv`) |
| Size | 19,158 rows × 14 columns (incl. `enrollee_id` and `target`) |
| Type | Structured / tabular |
| Target | `target`: 0 = not looking for a job change, 1 = looking for a job change |
| Challenges | Missing values, class imbalance, high-cardinality `city`, ordinal-like text fields (`experience`, `last_new_job`, `company_size`) |
| License | CC0: Public Domain (see `Dataset/dataset_info.txt`) |

## 5. Repository Structure (After Phase 1)

```
SW485-Project-Group#/
├── README.md
├── Phase1_Data_Exploration.ipynb
├── Phase1_Supervised_Learning.ipynb
├── Dataset/
│   ├── raw_kaggle_data.csv
│   ├── preprocessed_data.csv
│   └── dataset_info.txt
├── Supervised_Learning/
│   ├── models/               # saved model files (optional)
│   ├── evaluation_results/   # confusion matrices, metrics
│   └── plots/                # saved visualizations
└── docs/                     # task distribution, meeting notes, references
```

Phase 2 will add `Unsupervised_Learning/`, `Generative_AI/`, `Dataset/clustered_data.csv`, and the dashboard link.

## 6. Project Roadmap

| Phase | Deadline | Components | Status |
|-------|----------|-----------|--------|
| 1 | Oct 22, 2026 | Problem understanding & data exploration; supervised learning | In progress |
| 2 | Dec 3, 2026 | Unsupervised learning; structured prompt engineering & GenAI; interactive executive dashboard | Not started |

## 7. How to Run

```bash
git clone https://github.com/<org>/SW485-Project-Group#.git
cd SW485-Project-Group#
pip install -r requirements.txt
jupyter notebook
```
Run in order: `Phase1_Data_Exploration.ipynb` → `Phase1_Supervised_Learning.ipynb`.

## 8. Use of Assistive Tools

_<Per the handbook, state clearly how any AI tools were used, e.g., "used to improve the wording of the README; all analysis and code are our own.">_

## 9. Dashboard (Phase 2)

_Link to the executive interface will be added here in Phase 2._
