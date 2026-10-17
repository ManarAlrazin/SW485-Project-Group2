# Task Distribution

## Phase 1 (Deadline: Oct 22, 2026)

| Member | Responsibility | Deliverable | Internal Deadline |
|--------|----------------|-------------|-------------------|
| Manar Alrazin | Problem statement, README, repo setup, dataset justification, notebook intro, merging notebooks & final review | README.md, repo, dataset_info.txt, notebook sections 1-2, final QA | Oct 8 (setup), Oct 19-21 (review) |
| Noura Alaqeel | Data inspection, statistics, missing values handling, key insights | Notebook: Initial Data Inspection, Key Insights & Challenges | Oct 8 |
| Rana Albarrak | EDA visualizations + interpretation | Notebook: Exploratory Data Analysis | Oct 12 |
| Tojan Alnajjar | Encoding, scaling, outliers, feature engineering, train/test split, imbalance handling | Notebook: Preprocessing Pipeline, preprocessed_data.csv | Oct 12 |
| Munirah Alghayth | Model 1: rationale, training, tuning, feature importance, error analysis | Phase1_Supervised_Learning.ipynb (Model 1 sections) | Oct 15 |
| Ghala Alsanie | Model 2: rationale, training, tuning, error analysis | Phase1_Supervised_Learning.ipynb (Model 2 sections) | Oct 15 |
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
My part was the problem statement, the README, the repository setup, the dataset justification, and the final review.
The most important thing I learned is to read the project description carefully and write only what it asks for. My first draft of the introduction had extra sections (users, objectives, scope) that the description does not require, so I shortened it to a clear introduction and a concise problem statement.
I also learned why the team rules matter: dropping the ID column and splitting the data before scaling or encoding prevents data leakage, so I made sure every member knew them before starting.
Working with Git as a team was new to me: uploading folders through the website did not keep them, so I had to fix the structure by hand. Later I had to merge the notebooks from different members into the two required ones, and clean the commit history after everyone finished.
The hardest part was coordinating the order of work. Preprocessing depends on the cleaning, and the models depend on the preprocessing, so one late part delays the others.
### Noura Alaqeel
My part was data inspection, statistics, missing values handling and key insights.
The most important thing I learned is that missing values are not always just "empty" data. In our dataset, candidates who did not report their company were about twice as likely to look for a new job (38.8% vs 18.4%), so instead of deleting these rows or filling them with the most common value, I kept them as an "Unknown" category.
I also learned what data leakage means and why we should not use mean or mode imputation before the train/test split.
It was also my first time working with notebooks in VS Code, so setting up the environment took me some time.
The hardest part for me was deciding how to handle each column with missing values: when to fill it with "Unknown", when to use "No Major" and when to drop the rows.
### Rana Albarrak
Working on the EDA helped me understand the data better and see how each plot can give useful information for the model. One interesting finding was that missing values can actually be useful. For example, candidates with missing company information had a higher job-change rate. The main challenge was working in a shared notebook. I had to make sure I did not repeat Noura’s plots or change my teammates’ work. I used an AI assistant to help with the plotting code. Then I reviewed the results and wrote the interpretations based on my understanding.
### Tojan Alnajjar
My part was preprocessing: encoding, scaling, outliers, feature engineering, the train/test split, and the class-imbalance decision.

The most important thing I learned is that the order of the steps matters. I split the data first and fitted the imputer, encoder and scaler on the training set only, so no information from the test set leaked into the models. I also learned that a missing value can carry information: candidates with no company information looked for a new job about 42% of the time versus 18% for the others, so I turned this into a feature (company_info_missing) instead of just filling it.

Another thing I learned is why SMOTE should not be applied to the saved training file: synthetic rows would end up inside the cross-validation folds and inflate the scores. So I left the files without resampling and recommended class weights instead.

The hardest part was handling “Unknown” in the ordinal columns (company_size and education_level), because Unknown has no place on an ordered scale. I solved it with a median fill from the training set plus separate flag columns. I also had to coordinate with my teammates so Munira, Ghala and Fatimah could use the same train and test files.

I used an AI assistant to help write and explain the preprocessing code. I then ran it, checked the outputs, and wrote this reflection based on my own understanding of the steps.
### Munira Algayth
I built the Logistic Regression model, covering its rationale, tuning, evaluation, feature importance, and error analysis. My main takeaway was learning to handle class imbalance properly — using balanced class weights and judging the model on precision, recall, F1, and ROC-AUC instead of misleading accuracy. I also learned to interpret the model by turning coefficients into odds ratios to explain what actually drives job change. A useful challenge was understanding why tuning barely changed my linear model while it improved XGBoost, which taught me that simple, well-conditioned models have little to gain from tuning — and to present that result honestly rather than overstate it.
### Ghala Alsanie
I built and tuned the XGBoost Classifier, handling its setup, hyperparameter optimization, evaluation, and visualizations. My main takeaway was seeing how tree-based ensemble models strongly benefit from hyperparameter tuning using GridSearchCV to tune depth and learning rate directly boosted our ROC-AUC to 0.82 and increased Recall to 56.80%. I also learned how to extract and interpret feature importance from gradient boosting to identify the key factors driving employee attrition. Overcoming setup warnings and fine-tuning parameters taught me how non-linear models capture complex relationships that simpler models miss, providing actionable insights for HR retention strategies.
### Fatimah Omran
My part was model evaluation, comparison, conclusion and next steps for Phase 2.
The most important thing I learned is that accuracy is not enough when the data is imbalanced. XGBoost had higher accuracy, but it missed many of the people who were actually looking for a new job, so recall, F1 and ROC-AUC were more useful for comparing the models.
I also learned that the decision threshold can change the result a lot. At 0.5 the two models looked equal, but with a threshold chosen from the training data, XGBoost was clearly better.
The hardest part for me was understanding the tools and libraries I needed for the comparison. I spent time searching what each library does, like scikit-learn for the metrics and seaborn and matplotlib for the charts. I also had to learn how to show the confusion matrices and the ROC curves of both models side by side.
My part came after my teammates' parts, so I needed their models to be ready before I could finish the comparison. I was careful to only upload my own files so I would not affect their work.
