import os
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def setup_logger(log_file="mlflow_logging.log"):
    """Sets up Python logging to file and console."""
    logger = logging.getLogger("HandGestureLogger")
    
    # Avoid adding multiple handlers if function is called twice
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Write to log file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log to notebook console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger

def setup_mlflow_experiment(experiment_name, tracking_uri="http://localhost:5000"):
    """Sets tracking URI and active experiment."""
    os.environ["LOGNAME"] = "Ahmed Gamal"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp.experiment_id

def log_search_runs(search_model, model_name, phase, X_train, y_train, parent_metrics, parent_artifacts, logger):
    """Retroactively logs fitted SearchCV combinations as nested MLflow runs."""
    logger.info(f"Logging {model_name} search results for phase: {phase}")
    parent_name = f"{model_name}_Search"
    
    with mlflow.start_run(run_name=parent_name):
        # Set tags for tracking
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("phase", phase)
        
        # Log best params and best score from parent
        best_params = {f"best_{k}": v for k, v in search_model.best_params_.items()}
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", search_model.best_score_)
        mlflow.log_metrics(parent_metrics)
        
        # Log input training data
        train_df = X_train.copy()
        train_df["target"] = y_train
        dataset = mlflow.data.from_pandas(train_df, targets="target", name=f"{model_name}_Train")
        mlflow.log_input(dataset, context="Training")
        
        # Log artifacts for the parent (best model)
        for path in parent_artifacts:
            if os.path.exists(path):
                mlflow.log_artifact(path)
            else:
                logger.warning(f"Parent Artifact not found, skipping: {path}")
        
        # Log parent model architecture
        signature = infer_signature(X_train, search_model.predict(X_train))
        mlflow.sklearn.log_model(search_model.best_estimator_, f"{model_name}_best_estimator", signature=signature)
        
        # Iterate over all param combinations tried and log as nested child runs
        cv_results = search_model.cv_results_
        n_trials = len(cv_results["params"])
        logger.info(f"Logging {n_trials} child runs for {model_name}...")
        
        for i in range(n_trials):
            child_name = f"{model_name}_trial_{i+1}"
            with mlflow.start_run(run_name=child_name, nested=True):
                mlflow.set_tag("model", model_name)
                mlflow.set_tag("phase", phase)
                mlflow.set_tag("is_child", "true")
                
                # Log iteration-specific params and results
                mlflow.log_params(cv_results["params"][i])
                mlflow.log_metrics({
                    "mean_cv_score": cv_results["mean_test_score"][i],
                    "std_cv_score": cv_results["std_test_score"][i]
                })
                
def log_final_run(model, model_name, phase, X_test, y_test, metrics, artifacts, logger):
    """Logs the final winner model, metrics dict, and list of artifact paths."""
    logger.info(f"Logging final run for {model_name}")
    
    with mlflow.start_run(run_name=f"{model_name}_Final"):
        # Set final tags
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("phase", phase)
        
        # Log the dictionary of metrics directly
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        for path in artifacts:
            if os.path.exists(path):
                mlflow.log_artifact(path)
            else:
                logger.warning(f"Artifact not found, skipping: {path}")
                
        # Log input test data
        test_df = X_test.copy()
        test_df["target"] = y_test
        dataset = mlflow.data.from_pandas(test_df, targets="target", name=f"{model_name}_Test")
        mlflow.log_input(dataset, context="Testing")
        
        # Log final fitted model
        signature = infer_signature(X_test, y_test)
        mlflow.sklearn.log_model(
            model, 
            artifact_path="final_model", 
            signature=signature, 
            input_example=X_test.iloc[:2]
        )
