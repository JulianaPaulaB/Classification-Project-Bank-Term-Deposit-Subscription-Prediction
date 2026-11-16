# Bank Term Deposit Subscription Prediction
### A Classification Project Using the Bank Marketing Dataset
 
A data mining project that investigates which customer, campaign, and economic characteristics are associated with subscribing to a bank term deposit, and compares four classification models for predicting whether a contacted customer will subscribe.
 
Developed as part of Analytics Application: Finals Dossier Two (Classification Machine Learning).
 
---
 
## 📌 Project Overview
 
Banks run telephone marketing campaigns to promote term deposits, but most of the customers they contact say no. Each call costs staff time and money, so the bank needs to know which customers are more likely to subscribe and how to prioritize its efforts.
 
This project uses 41,176 campaign records from a Portuguese bank to examine those patterns and to build a classification model that predicts whether a customer will subscribe (`yes` or `no`). Four classifiers are trained, tested on unseen records, and compared using accuracy, precision, recall, and F1-score for the `yes` class.
 
---
 
## 🎯 Project Objectives
 
- Understand customer and campaign characteristics associated with term deposit subscriptions.
- Support more efficient use of marketing resources by identifying subscription patterns.
- Develop classification models that predict whether a customer will subscribe.
- Compare Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest models.
- Select the best model using metrics suited to an imbalanced target.
- Save the best model and use it to generate predictions.
---
 
## 📊 Dataset
 
**Source:** [UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
**Original research:** Moro, Cortez, and Rita (2014)
**File used:** `bank-additional-full.csv` (semicolon-separated)
**Licensing:** See the dataset's UCI page for current licensing details.
 
| Group | Attributes |
|---|---|
| Customer | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Last contact | `contact`, `month`, `day_of_week`, `duration` |
| Campaign history | `campaign`, `pdays`, `previous`, `poutcome` |
| Economic indicators | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |
| Target | `y` – subscribed to a term deposit (`yes` / `no`) |
 
The raw file has 41,188 records and 21 columns, with no missing values. Some categorical columns contain an `unknown` category.
 
### Class Imbalance
 
| Outcome | Percentage |
|---|---|
| no | 88.73% |
| yes | 11.27% |
 
Because most customers did not subscribe, a model that always predicts `no` would already reach about 88.7% accuracy. For this reason, the models are compared mainly on precision, recall, and F1-score for the `yes` class.
 
---
 
## 🛠️ Tools and Technologies
 
- **Python** – Data processing and modeling
- **Pandas & NumPy** – Data cleaning and manipulation
- **Scikit-learn** – Preprocessing pipelines, classifiers, and evaluation metrics
- **Matplotlib** – Data visualization
- **Joblib** – Saving and loading the best model
- **Jupyter Notebook** – Analysis and model development
---
 
## 📈 Analytical Methodology
 
1. **Business and Data Understanding** – Defined the business context, problem, objectives, assumptions, and success criteria. Documented the dataset metadata and checked the data types, missing values, duplicates, and value ranges.
2. **Exploratory Data Analysis** – Examined the target distribution, customer ages, subscription rates by job, correlations among numerical features, and numeric edge cases.
3. **Data Partitioning** – Removed 12 exact duplicate records, then split the remaining 41,176 records 80% training (32,940) and 20% testing (8,236) with `random_state=42`. The split is stratified, so both sets keep the same 11.27% `yes` rate.
4. **Pre-processing** – Built one preprocessing pipeline that is fitted on the training records only.
5. **Pipeline and Model Training** – Trained four classifiers, each in a pipeline that includes the preprocessing steps.
6. **Model Comparison** – Compared the models on the testing set and selected the best one by the F1-score of the `yes` class.
7. **Save, Load, and Predict** – Saved the best model, reloaded it, and generated predictions for the testing records.
### Excluding `duration`
 
`duration` (the length of the last call) is known only after the call ends, so it cannot be used to decide whom to call. It is removed from the model inputs, which leaves 19 predictors.
 
### Preprocessing Pipeline
 
| Feature Type | Count | Processing |
|---|---|---|
| Numerical (`age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) | 9 | Median imputation, then standardization |
| Categorical (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`) | 10 | Most-frequent imputation, then one-hot encoding |
 
After preprocessing, the data has 62 model-ready features.
 
### Models Compared
 
| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| K-Nearest Neighbors | `n_neighbors=5` |
| Decision Tree | `class_weight="balanced"` |
| Random Forest | `n_estimators=100`, `class_weight="balanced"` |
 
---
 
## 🏆 Model Results
 
Evaluated on the 8,236 testing records, ordered by the F1-score of the `yes` class.
 
| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) |
|---|---|---|---|---|
| **Logistic Regression** | 0.8303 | 0.3591 | **0.6455** | **0.4615** |
| K-Nearest Neighbors | 0.8955 | 0.5661 | 0.3093 | 0.4000 |
| Random Forest | 0.8925 | 0.5464 | 0.2726 | 0.3638 |
| Decision Tree | 0.8458 | 0.3237 | 0.3384 | 0.3309 |
 
### Confusion Matrix of the Best Model (Logistic Regression)
 
| | Predicted no | Predicted yes |
|---|---|---|
| **Actual no** | 6,239 | 1,069 |
| **Actual yes** | 329 | 599 |
 
---
 
## 🔍 Key Findings
 
- **Logistic Regression performed best.** It has the highest F1-score (0.4615) and by far the highest recall (0.6455), correctly identifying 599 of the 928 customers who actually subscribed.
- **Accuracy alone is misleading here.** K-Nearest Neighbors and Random Forest have the highest accuracy (about 0.89), but this is close to the 88.7% reached by always predicting `no`, and they find only about 27–31% of the real subscribers.
- **There is a precision and recall trade-off.** Logistic Regression flagged 1,668 customers as likely subscribers, and 599 of them (35.9%) did subscribe. That is about 3.2 times the overall subscription rate of 11.3%, but it also means most flagged customers still do not subscribe.
- **Practical meaning.** In the testing set, contacting only the customers flagged by the model (about 20% of the 8,236) would have reached about 65% of the actual subscribers. The model can help prioritize calls, but it does not guarantee what an individual customer will do.
- **The Decision Tree was the weakest model** by F1-score (0.3309).
---
 
## 📁 Project Structure
 
```
bank-marketing-term-deposit-prediction/
│
├── FinalsDossier2_Biñas_JulianaPaula.ipynb
├── bank-additional-full.csv
├── best_bank_marketing_model.pkl
├── requirements.txt
└── README.md
```
 
---
 
## 📓 Notebook Contents
 
| Part | Section |
|---|---|
| 1–2 | Dataset selection and Business Understanding |
| 3 | Data Understanding (metadata, description, data quality, exploration) |
| 3 | Data Partitioning |
| 4 | Data Pre-processing |
| 5 | Machine Learning Pipeline (5.1 Logistic Regression, 5.2 K-Nearest Neighbors, 5.3 Decision Tree, 5.4 Random Forest) |
| 6 | Model Comparison |
| 7 | Save the Best Model |
| 8 | Load the Best Model |
| 9 | Create Predictions from the X_test Data |
| 10 | Show X_test with the Actual and Predicted Outcomes |
 
---
 
## 🚀 How to Run the Project Locally
 
### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/bank-marketing-term-deposit-prediction.git
```
 
### 2. Navigate to the Project Folder
```bash
cd bank-marketing-term-deposit-prediction
```
 
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
 
### 4. Open the Notebook
```bash
jupyter notebook FinalsDossier2_Biñas_JulianaPaula.ipynb
```
 
Then choose **Restart and Run All**.
 
### requirements.txt
```
pandas
numpy
scikit-learn==1.6.1
matplotlib
joblib
jupyter
```
 
The saved model was created with scikit-learn 1.6.1. Newer versions may fail to load `best_bank_marketing_model.pkl`, and results from the Decision Tree and Random Forest can differ slightly, so the version is pinned.
 
### Using the Saved Model
 
```python
import joblib
import pandas as pd
 
model = joblib.load("best_bank_marketing_model.pkl")
 
# The model expects the 19 predictors (no "y" and no "duration")
df = pd.read_csv("bank-additional-full.csv", sep=";")
X_new = df.drop(columns=["y", "duration"]).head()
 
print(model.predict(X_new))
```
 
---
 
## 💼 Business Relevance
 
Bank management, marketing managers, and campaign planners need to decide whom to contact and how to spend limited calling resources. This project shows how historical campaign records can be used to estimate which customers are more likely to subscribe to a term deposit, so that outreach can be planned with data and not only with judgment.
 
---
 
## ⚠️ Limitations
 
- The target is imbalanced (11.27% `yes`), so accuracy should not be used on its own to judge the models.
- The scores come from a single 80/20 split. The best model was chosen from the testing-set results, and no separate validation set or cross-validation was used, so the selected score may be slightly optimistic.
- Hyperparameters were left at simple defaults and were not tuned, and the default 0.5 decision threshold was used.
- Precision of the best model is 0.36, so most customers it flags as likely subscribers do not subscribe.
- The data comes from the historical campaigns of one Portuguese bank and may not represent other banks, markets, or current customers.
- `duration` was excluded because it is known only after a call, so the model uses information available before contacting a customer.
- The results show associations in historical records. They do not show that any attribute causes a customer to subscribe.
---
 
## 👩‍💻 Author
 
**Juliana Paula T. Biñas**
BS Information Systems – Business Analytics
West Visayas State University
 
Academic Project | Analytics Application: Finals Dossier Two
 
*Turning data into insights for smarter marketing decisions.*
 
