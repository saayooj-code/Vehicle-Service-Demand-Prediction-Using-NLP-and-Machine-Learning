# Vehicle-Service-Demand-Prediction-Using-NLP-and-Machine-Learning
Hybrid NLP and Machine Learning project that analyzes customer complaints, vehicle information, service history, and workshop conditions to predict service demand, workshop load, technicians required, and service bays. Uses TF-IDF and XGBoost with an interactive Gradio interface.
# 🚗 Vehicle Service Demand Prediction Using NLP and Machine Learning

A hybrid **Natural Language Processing (NLP) + Machine Learning** project that analyzes customer complaints, vehicle information, service history, and workshop conditions to predict **vehicle service demand and workshop resource requirements**.

The system uses **TF-IDF for NLP feature extraction** and **XGBoost regression models** to generate multiple predictions through an interactive **Gradio interface**.

---

## 📌 Project Overview

Vehicle service centers receive different types of customer complaints and service requests every day. Manually analyzing this information can make it difficult to estimate upcoming service demand and prepare the required workshop resources.

This project combines:

* 📝 Customer service-related text
* 🚘 Vehicle information
* 🔧 Previous service and repair history
* 🏭 Workshop conditions
* 📅 Time and seasonal information
* 📊 Local demand and appointment backlog

The system predicts:

1. **Service Demand**
2. **Workshop Load Forecast**
3. **Technicians Required**
4. **Service Bays Required**

It also provides demand and parts-requirement levels with a workshop recommendation.

---

## 🎯 Objectives

* Analyze vehicle service-related text using NLP.
* Extract useful information from multiple text inputs.
* Combine NLP features with numerical and categorical data.
* Predict upcoming vehicle service demand.
* Forecast workshop workload.
* Estimate the number of technicians required.
* Estimate the number of service bays required.
* Provide an interactive prediction interface.
* Generate a practical workshop recommendation.

---

## 🧠 NLP Features

The project processes **five different text inputs**:

| NLP Feature            | Description                                |
| ---------------------- | ------------------------------------------ |
| `Customer_Complaint`   | Customer's reported vehicle problem        |
| `Vehicle_Symptoms`     | Symptoms observed in the vehicle           |
| `Service_History_Text` | Previous service information               |
| `Technician_Notes`     | Technician observations                    |
| `Customer_Concern`     | Customer's specific concern or requirement |

### NLP Processing

The text is processed using:

* Lowercase conversion
* Special-character removal
* Whitespace normalization
* TF-IDF vectorization
* Unigram and bigram features

Each text feature has its own TF-IDF vectorizer with a maximum of **1,000 features**.

---

## 🚘 Vehicle Features

The model also uses vehicle-related information such as:

* Vehicle Type
* Vehicle Age
* Mileage
* Engine Type
* Service Category
* Previous Service Count
* Previous Repair Count
* Average Service Gap

---

## 🏭 Workshop Features

Workshop conditions used by the model include:

* Parts Availability
* Technician Availability
* Workshop Load
* Bay Utilization
* Local Demand Index
* Appointment Backlog

---

## 📅 Time & Customer Features

The project also considers:

* Season
* Day of Week
* Weekend Indicator
* Customer Segment
* Workshop Zone

---

## 🎯 Prediction Targets

The project trains separate XGBoost regression models for four targets:

```text
Service_Demand
Workshop_Load_Forecast
Technicians_Required
Service_Bays_Required
```

This makes the project a **multi-output business decision-support system**, rather than predicting only one value.

---

## 🔄 Machine Learning Workflow

```text
Vehicle Service Dataset
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Categorical Encoding
        ↓
Numerical Feature Processing
        ↓
Feature Combination
        ↓
Train / Test Split
        ↓
XGBoost Regression
        ↓
Model Evaluation
        ↓
Multiple Predictions
        ↓
Workshop Recommendation
```

---

## ⚙️ Data Preprocessing

### Missing Values

Missing values are handled differently according to feature type:

* Text features → empty string
* Categorical features → mode
* Numerical features → median

### Categorical Encoding

Categorical features are converted using:

```python
OneHotEncoder(handle_unknown="ignore")
```

### Feature Combination

The final model input combines:

```text
TF-IDF Features
+
One-Hot Encoded Features
+
Numerical Features
```

Sparse matrices are combined using SciPy's `hstack()`.

---

## 🤖 Machine Learning Model

### XGBoost Regressor

The project uses **XGBRegressor** for all four prediction targets.

Main configuration includes:

```text
n_estimators = 250
max_depth = 6
learning_rate = 0.08
subsample = 0.8
colsample_bytree = 0.8
objective = reg:squarederror
```

Four separate models are trained, one for each target.

---

## 📊 Model Evaluation

The project evaluates each prediction model using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

### RMSE — Root Mean Squared Error

Measures prediction error while giving more weight to larger errors.

### R² Score

Measures how well the model explains the variation in the target variable.

The notebook calculates these metrics separately for all four prediction targets.

> **Note:** The repository should use the actual metric values produced by your final notebook run rather than manually adding or estimating scores.

---

## 💾 Saved Model Files

After training, the project saves:

```text
vehicle_service_models.pkl
vehicle_service_tfidf.pkl
vehicle_service_encoder.pkl
vehicle_service_features.pkl
```

### File Description

| File                           | Purpose                                |
| ------------------------------ | -------------------------------------- |
| `vehicle_service_models.pkl`   | Stores the four trained XGBoost models |
| `vehicle_service_tfidf.pkl`    | Stores the TF-IDF vectorizers          |
| `vehicle_service_encoder.pkl`  | Stores the categorical encoder         |
| `vehicle_service_features.pkl` | Stores feature configuration           |

---

## 🖥️ Gradio Interface

The project includes an interactive **Gradio web interface** called:

```text
🚗 Vehicle Service AI Predictor
```

Users can enter service-related text, vehicle information, workshop conditions, and customer information.

### Input Sections

#### 📝 NLP Inputs

* Customer Complaint
* Vehicle Symptoms
* Service History
* Technician Notes
* Customer Concern

#### 🚘 Vehicle Information

* Vehicle Type
* Vehicle Age
* Mileage
* Engine Type
* Service Category
* Previous Service Count
* Previous Repair Count
* Average Service Gap

#### 🏭 Workshop Information

* Parts Availability
* Technician Availability
* Current Workshop Load
* Bay Utilization
* Local Demand Index
* Appointment Backlog

#### 📅 Time & Customer

* Season
* Day of Week
* Weekend Indicator
* Customer Segment
* Workshop Zone

---

## 📈 Prediction Outputs

The interface provides:

```text
Service Demand
Workshop Load Forecast
Technicians Required
Service Bays Required
Demand Level
Parts Requirement
Workshop Recommendation
```

Example recommendation format:

```text
Prepare the workshop for approximately X vehicles.
Arrange X technicians and X service bays.
Expected workshop load is X%.
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TF-IDF**
* **SciPy**
* **XGBoost**
* **Joblib**
* **Gradio**
* **Jupyter Notebook / Anaconda**
