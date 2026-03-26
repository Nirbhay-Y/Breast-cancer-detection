# 🧬 Breast Cancer Detection using Machine Learning

## 📌 Overview

This project is an **end-to-end Machine Learning application** that predicts whether a breast tumor is **Benign** or **Malignant** using diagnostic features.
It demonstrates the **full ML lifecycle** — from feature selection and model training to inference and deployment via a Streamlit web app.

The focus of this project is not just accuracy, but **clean ML engineering practices**, interpretability, and reproducibility.

---

## 🧠 Problem Statement

Early detection of breast cancer is critical for effective treatment.
The goal of this project is to build a machine learning model that can assist in **risk assessment** by analyzing tumor characteristics extracted from medical imaging.

⚠️ **Disclaimer**:
This application is for **educational purposes only** and should **not** be considered a medical diagnosis tool.

---

## 📊 Dataset

* **Source**: Breast Cancer Wisconsin Dataset
* **Provider**: `sklearn.datasets`
* **Samples**: 569
* **Features**: 30 numerical diagnostic features
* **Target**:

  * `0` → Malignant
  * `1` → Benign

The dataset is loaded programmatically using `sklearn.datasets`, so no external data files are required.

---

## ⚙️ ML Pipeline

### 1️⃣ Feature Selection

* **Technique**: Mutual Information (`mutual_info_classif`)
* **Approach**:

  * Rank all features based on information gain
  * Select **Top 10 most informative features**
* **Reason**:

  * Improves interpretability
  * Reduces noise
  * Maintains strong performance

---

### 2️⃣ Data Preprocessing

* Train–test split (80/20)
* Feature scaling using **StandardScaler**
* Preprocessing logic is centralized to ensure **training–inference consistency**

---

### 3️⃣ Model Training

* **Model**: Logistic Regression
* **Why Logistic Regression?**

  * Interpretable
  * Stable
  * Well-suited for medical risk prediction
* Model artifacts saved:

  * Trained model
  * Scaler
  * Selected feature list

---

### 4️⃣ Inference

* A single inference module handles:

  * Column alignment
  * Scaling
  * Prediction
  * Probability estimation
* Prevents feature mismatch errors and logic duplication

---

## 🖥️ Web Application (Streamlit)

The Streamlit app provides:

* User-friendly input form with guided placeholders
* Prediction output (Benign / Malignant)
* Confidence score (prediction probability)
* Graceful input validation and error handling

The UI is intentionally kept **lightweight**, while all ML logic resides in the backend pipeline.

---

## 📁 Project Structure

```text
breast-cancer-ml/
│
├── notebooks/
│   └── breast_cancer_eda.ipynb
│
├── src/
│   ├── feature_selection.py
│   ├── preprocessing.py
│   ├── train.py
│   └── inference.py
│
├── models/
│   ├── reduced_LR_model.pkl
│   ├── reduced_scaler_LR.pkl
│   └── reduced_columns_LR.pkl
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model

```bash
python src/train.py
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Model Behavior Notes

* The model outputs **prediction probabilities**
* Borderline cases may lean towards **Malignant**, prioritizing safety
* This behavior is intentional and appropriate for medical screening contexts

---

## 🚀 Future Improvements

* Add model explainability using **SHAP**
* Experiment with regularization tuning
* Extend to multi-model comparison dashboard
* Deploy as a cloud-based application

---

## 🎯 Key Learnings

* Feature selection is as important as model choice
* Clean separation of training and inference logic prevents real-world bugs
* Simple models, when engineered well, can be highly effective
* ML engineering is about **systems**, not just algorithms

---

## 👤 Author

**Nirbhay Yadav**
CSE (AIML) Undergraduate | Aspiring Machine Learning Engineer

---

## ⭐ Acknowledgements

* Scikit-learn
* Streamlit
* Breast Cancer Wisconsin Dataset