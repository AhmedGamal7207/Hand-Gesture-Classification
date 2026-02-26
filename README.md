# Hand Gesture Classification
A project that will implement a Hand Gesture Classification on HaGRID Dataset. It gets the landmarks from mediapipe and classify the gesture using a trained classifier.

This repository has 2 branches (main and research)
main branch: is for only the hand gesture classification project code
research: same as main but with MlFlow code

---

## 🔬 MLflow Tracking Architecture

To maintain a clean, professional, and highly organized experimental tracking environment, this project implements a rigorous **Parent-Child tracking hierarchy** via a custom `mlflow_logging` module.

> This architecture eliminates UI clutter while preserving deep, granular data for every hyperparameter combination tested, allowing us to rapidly iterate and compare multiple models.

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
