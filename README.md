# Hand Gesture Classification
A project that will implement a Hand Gesture Classification on HaGRID Dataset. It gets the landmarks from mediapipe and classify the gesture using a trained classifier.

This repository has 2 branches (main and research)
main branch: is for only the hand gesture classification project code
research: same as main but with MlFlow code

---

## 🔬 MLflow Tracking Architecture

To maintain a clean, professional, and highly organized experimental tracking environment, this project implements a rigorous **Parent-Child tracking hierarchy** via a custom `mlflow_logging` module.

> This architecture eliminates UI clutter while preserving deep, granular data for every hyperparameter combination tested, allowing us to rapidly iterate and compare multiple models.

### 📈 Tracking Statistics
* **Total MLflow Runs Tracked:** `535`
* **Parent Runs (Algorithm Evaluations):** `28`
* **Child Runs (Hyperparameter Variations):** `507`

### 🏗️ Hierarchy Design

| Level | Component | Description | Tags |
| :---: | :--- | :--- | :--- |
| **1️⃣** | **Experiment** | The global container for all classification trials. | `Hand_Gestures_Classification` |
| **2️⃣** | **Parent Run** | Represents an abstract Algorithm evaluation (e.g., `RandomForest_Search`). Stores the *best model artifact*, overarching metrics, and data signatures. | `model`, `phase: qualifications` |
| **3️⃣** | **Child Runs** | Nested trials under their respective Parent Run (e.g., `trial_1`, `trial_2`). Each represents a specific *hyperparameter combination*. | `is_child: true`, `model`, `phase` |

<br>

### 🚀 Pipeline Phases

We manage the lifecycle of our models across three distinct deployment phases:

#### 1. `🔍 Phase: Qualifications`
* Broad randomized hyperparameter searches (`RandomizedSearchCV`) across our baseline models: `LightGBM`, `XGBoost`, `RandomForest`, and `LogisticRegression`.
* **Architecture:** **Parent** *(Algorithm Summary)* ➔ **Children** *(Parameter Variations)*

#### 2. `🎯 Phase: Fine-Tuning`
* High-resolution parameter sweeps (`GridSearchCV`) targeting the top 3 high-potential models identified during Qualifications. 
* **Architecture:** **Parent** *(Detailed Best Model)* ➔ **Children** *(Precise Parameter Variations)*

#### 3. `🏆 Phase: Final`
* The single undisputed champion model, evaluated on strictly held-out `X_test` data.
* Tracks ultimate `f1_macro` metrics, combined with serialized artifacts (`.pkl`), the `Confusion Matrix`, and `ROC Curves`.
* **Architecture:** Standalone run & registration to the MLflow Model Registry.

<br>

## 📊 Pipeline Results & Model Comparison 

By tracking each phase meticulously via MLflow, we captured the progression of our classifiers from baseline combinations to highly optimized champion architectures. The comparison below illustrates this journey.

### 1. Qualifications Phase (Randomized Baseline)
We initially evaluated four distinct algorithm concepts using 3-fold Stratified Cross-Validation to quickly establish performance ceilings.

| Model Algorithm | Type | Best CV F1-Macro | 
| :--- | :--- | :---: |
| **LightGBM** | Light Gradient Boosting Tree | **0.9794** | 
| **XGBoost** | Extreme Gradient Boosting Tree | **0.9759** | 
| **RandomForest** | Ensembled Bagging Tree | 0.9652 | 
| **LogisticRegression**| Linear Baseline | 0.9481 | 

*The tree-based boosted learners generalized best over the high-dimensional landmark coordinates. LightGBM and XGBoost were formally selected to advance.*

<br>

### 2. Fine-Tuning Phase (Exhaustive Grid Search)
Our top two models were subjected to rigorous hyperparameter tuning to mathematically optimize components like tree depth, learning rate, and estimators. Both models pushed beyond their initial baselines.

| Model | Pre-Tuning F1 | Post-Tuning F1 | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **LightGBM** | 0.9794 | **0.9802** | + 0.0008 |
| **XGBoost** | 0.9759 | **0.9792** | + 0.0033 |

#### 🔑 The Winning Hyperparameter Architectures
* **LightGBM:** `n_estimators`: 400 \| `learning_rate`: 0.1 \| `num_leaves`: 31 \| `subsample`: 0.8 \| `min_child_samples`: 20
* **XGBoost:** `n_estimators`: 400 \| `learning_rate`: 0.1 \| `max_depth`: 7 \| `subsample`: 0.8 \| `reg_lambda`: 5.0

<br>

**Visualizing the Fine-Tuned Validation Patterns:**
<p align="center">
  <img src="plots_finals/Confusion_Matrix_Validation_Fine_Tuning_LightGBM.png" width="48%" />
  <img src="plots_finals/Classification_Report_Validation_Fine_Tuning_LightGBM.png" width="48%" />
</p>
<p align="center"><em>LightGBM's remarkable internal structure processing the Validation splits after Tuning.</em></p>

<br>

### 3. Last Model Standing (Target Set Benchmark)
We pitted the two structurally finalized models against the pure, purely untouched `X_test` dataset to guarantee no variance leaked during cross-validation, allowing us to crown the ultimate algorithm.

| Rank | Model | Final Test F1-Macro | 
| :---: | :--- | :---: |
| 🥇 | **LightGBM (Tuned)** | **0.9843** |
| 🥈 | **XGBoost (Tuned)** | 0.9814 |

#### 🏆 Ultimate Champion: LightGBM
By securing an astounding **98.43% F1-Macro** over 18 disparate gesture classes using pure semantic landmarks, LightGBM is undeniably the most efficient and robust classification model. Its sheer lightweight nature theoretically minimizes API latency in production over XGBoost, too.

<p align="center">
  <img src="plots_last_standing/Confusion_Matrix_Testing_LightGBM.png" width="48%" />
  <img src="plots_last_standing/Classification_Report_Testing_LightGBM.png" width="48%" />
</p>
