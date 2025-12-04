This README is in English. For the French version, see [README.md](README.md).

# Diabetes Risk Prediction — Intelligent Biomedical System

## Project Overview

This project aims to develop an intelligent system capable of predicting whether a patient is at **high risk of developing diabetes**, using clinical features such as:

- **Glucose**
- **Blood Pressure**
- **Skin Thickness**
- **Insulin**
- **BMI** (Body Mass Index)
- **Diabetes Pedigree Function**
- **Age**

The system includes:

1. **Supervised Classification** — Identify patients with high or low diabetes risk.  
2. **Unsupervised Clustering** — Group patients based on biomedical similarity.



---

## Technologies Used

- **Python 3** — main language for data analysis and model development
- **Pandas / NumPy** — data manipulation and processing
- **Matplotlib / Seaborn** — statistical visualizations
- **Scikit-learn** — preprocessing, classification models, clustering, pipelines
- **XGBoost** — gradient boosting classifier
- **Joblib** — model serialization
- **Streamlit** — interactive user interface for real-time diabetes risk prediction

---

## Project Structure

```
📁 Diabetes-Risk-Prediction
│
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 README_en.md
├── 📄 main.py
├── 📁 data/
│    ├── 📁 raw/
│    └── 📁 processed/
├── 📁 models/
│    ├── 📄 model.pkl
│    └── 📄 scaler.pkl
├── 📁 notebooks/
│    ├── 📄 EDA.ipynb
│    ├── 📄 preprocessing.ipynb
│    ├── 📄 Clustering_KMeans.ipynb
│    ├── 📄 Classification.ipynb
│    └── 📄 Model_Evaluation.ipynb
```

---

## Running the Project

1. Clone the repository:
```bash
git lone https://github.com/anass17/Analyse-et-Prediction-du-Risque-de-Diabete
cd Analyse-et-Prediction-du-Risque-de-Diabete
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run main.py
```

4. Open the app in your browser:
`http://localhost:8501/`

---

## User Stories & Tasks

### User Story 1 : Data Loading & Exploratory Data Analysis (EDA)

- Load the dataset with **Pandas**
- Explore data structure and summary statistics
- Detect missing values and duplicates
- Analyze variable distributions
- Visualize correlations

---

### User Story 2 : Data Preprocessing

- Handle missing data
- Detect and manage outliers **boxplot, z-score, IQR**
- Select relevant features
- Normalize/standardize numerical columns
- Visualize variable relationships (pairplots)


---

### User Story 3 : K-Means Clustering

- Determine optimal **k** (Elbow & Silhouette methods)
- Train **K-Means** with best k
- Add `cluster` labels
- Visualize cluster distribution


---

### User Story 4 : Cluster Analysis

- Compute feature means per cluster
- Identify number of samples per cluster
- Detect high-risk clusters
- Add `risk_category` column (1 = high risk, 0 = low)

---

### User Story 5 : Supervised Classification & Model Evaluation

- Create target variable from clusters
- Split dataset (80% train / 20% test)
- Handle class imbalance
- Train models:
    - Random Forest
    - SVM
    - Gradient Boosting
    - Decision Tree
    - Logistic Regression
    - XGBoost
- Evaluate (confusion matrix, precision, recall, F1-score)
- Perform cross-validation
- Hyperparameter tuning
- Save best model



--- 

### Model Performance

| Model                | Accuracy           |
| ---------------------| ------------------ |
| Random Forest        | 0.9060402684563759 |
| XGB                  | 0.9530201342281879 |
| Gradient Boosting    | 0.9530201342281879 |
| Decision Tree        | 0.8590604026845637 |
| SVM                  | 0.9798657718120806 |
| Logistic Regression  | 0.9865771812080537 |

**Final Selected Model → Logistic Regression**

### Final Model

Saved at:

```
models/model.pkl
```

Load with:

```Python
import joblib
model = joblib.load("models/model.pkl")
```

---

## Visualisations

### Data distribution
![Initial data](https://github.com/user-attachments/assets/a3b30faf-5c5a-4f3a-b41d-32e036b954f7)
![Data after preprocessing](https://github.com/user-attachments/assets/f740bc4a-d4f7-4f03-b302-c420638d3570)

### Streamlit Interface 
![Streamlit UI](https://github.com/user-attachments/assets/2828820c-2b72-4ccf-9046-919f2453c5b7)

---