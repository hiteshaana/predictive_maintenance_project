Predictive Maintenance - Model Training Pipeline

Final MLOps pipeline:
    1. Load train/test directly from Hugging Face Dataset Hub
    2. Benchmark permitted classification models
    3. Tune Random Forest using GridSearchCV
    4. Use 5-fold cross-validation and F1 scoring
    5. Evaluate the optimized model
    6. Generate evaluation artifacts
    7. Log parameters, metrics and artifacts to MLflow
    8. Save the final model locally

The Hugging Face model-registration step is kept separate in
model_registry.py so that the training pipeline can be independently
executed and subsequently automated through GitHub Actions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(
    "predictive_maintenance_project"
)

DATA_DIR = PROJECT_DIR / "data"

MODEL_DIR = PROJECT_DIR / "model"

MLRUNS_DIR = PROJECT_DIR / "mlruns"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MLRUNS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TARGET = "Engine_Condition"

HF_USERNAME = os.getenv(
    "HF_USERNAME",
    "hiteshsharma"
)

HF_DATASET_REPO = os.getenv(
    "HF_DATASET_REPO",
    f"{HF_USERNAME}/predictive_maintenance-dataset"
)

HF_TRAIN_URI = (
    f"hf://datasets/"
    f"{HF_DATASET_REPO}/train.csv"
)

HF_TEST_URI = (
    f"hf://datasets/"
    f"{HF_DATASET_REPO}/test.csv"
)

RANDOM_STATE = 42


# ==========================================================
# LOAD TRAIN / TEST DIRECTLY FROM HUGGING FACE
# ==========================================================

def load_train_test_from_huggingface():

    print("=" * 80)
    print("LOADING TRAIN / TEST DATA FROM HUGGING FACE")
    print("=" * 80)

    print(
        f"Train source: {HF_TRAIN_URI}"
    )

    print(
        f"Test source : {HF_TEST_URI}"
    )

    train_df = pd.read_csv(
        HF_TRAIN_URI
    )

    test_df = pd.read_csv(
        HF_TEST_URI
    )

    print(
        f"\nTrain shape: {train_df.shape}"
    )

    print(
        f"Test shape : {test_df.shape}"
    )

    return train_df, test_df


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model
            .predict_proba(X_test)[:, 1]
        )

    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
    }

    if probabilities is not None:

        metrics["ROC_AUC"] = (
            roc_auc_score(
                y_test,
                probabilities
            )
        )

    return (
        metrics,
        predictions,
        probabilities
    )


# ==========================================================
# BASELINE MODEL COMPARISON
# ==========================================================

def benchmark_models(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON")
    print("=" * 80)

    models = {

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            ),

        "AdaBoost":
            AdaBoostClassifier(
                random_state=RANDOM_STATE
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),
    }

    results = []

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        metrics, _, _ = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append(
            {
                "Model": name,
                **metrics
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by=[
            "F1",
            "Recall",
            "ROC_AUC"
        ],
        ascending=False
    )

    results_path = (
        MODEL_DIR /
        "baseline_model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        "\n" + results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved baseline results to:"
        f"\n{results_path}"
    )

    return results_df


# ==========================================================
# RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================================

def tune_random_forest(
    X_train,
    y_train
):

    print("\n" + "=" * 80)
    print("RANDOM FOREST GRIDSEARCHCV")
    print("=" * 80)

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    param_grid = {

        "n_estimators": [
            100,
            200
        ],

        "max_depth": [
            None,
            10,
            15
        ],

        "min_samples_split": [
            2,
            5
        ],

        "min_samples_leaf": [
            1,
            2
        ],

        "max_features": [
            "sqrt",
            "log2"
        ],
    }

    grid = GridSearchCV(

        estimator=rf,

        param_grid=param_grid,

        scoring="f1",

        cv=5,

        n_jobs=-1,

        verbose=1,

        return_train_score=True
    )

    print(
        "\nStarting GridSearchCV..."
    )

    grid.fit(
        X_train,
        y_train
    )

    best_model = (
        grid.best_estimator_
    )

    print(
        "\nBest parameters:"
    )

    print(
        json.dumps(
            grid.best_params_,
            indent=4,
            default=str
        )
    )

    # ------------------------------------------------------
    # Save complete GridSearch results
    # ------------------------------------------------------

    cv_results = pd.DataFrame(
        grid.cv_results_
    )

    cv_results_path = (
        MODEL_DIR /
        "gridsearch_results.csv"
    )

    cv_results.to_csv(
        cv_results_path,
        index=False
    )

    # ------------------------------------------------------
    # Save best parameters
    # ------------------------------------------------------

    params_path = (
        MODEL_DIR /
        "best_parameters.json"
    )

    with open(
        params_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            grid.best_params_,
            f,
            indent=4,
            default=str
        )

    print(
        f"\nGridSearch results saved to:"
        f"\n{cv_results_path}"
    )

    print(
        f"\nBest parameters saved to:"
        f"\n{params_path}"
    )

    return (
        best_model,
        grid.best_params_
    )


# ==========================================================
# FINAL MODEL EVALUATION
# ==========================================================

def final_model_evaluation(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 80)
    print("FINAL MODEL EVALUATION")
    print("=" * 80)

    # ------------------------------------------------------
    # Train final optimized model
    # ------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    metrics, predictions, probabilities = (
        evaluate_model(
            model,
            X_test,
            y_test
        )
    )

    print(
        "\nFinal model metrics:"
    )

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    # ------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------

    metrics_path = (
        MODEL_DIR /
        "final_model_metrics.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_path,
        index=False
    )

    # ------------------------------------------------------
    # Classification report
    # ------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_path = (
        MODEL_DIR /
        "classification_report.csv"
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        report_path
    )

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Healthy",
            "Maintenance Required"
        ]
    ).plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        "Confusion Matrix - Optimized Random Forest"
    )

    fig.tight_layout()

    cm_path = (
        MODEL_DIR /
        "confusion_matrix.png"
    )

    fig.savefig(
        cm_path,
        dpi=220
    )

    plt.close(fig)

    # ------------------------------------------------------
    # ROC Curve
    # ------------------------------------------------------

    if probabilities is not None:

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            ax=ax
        )

        ax.set_title(
            "ROC Curve - Optimized Random Forest"
        )

        fig.tight_layout()

        roc_path = (
            MODEL_DIR /
            "roc_curve.png"
        )

        fig.savefig(
            roc_path,
            dpi=220
        )

        plt.close(fig)

    # ------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance = pd.Series(

            model.feature_importances_,

            index=X_train.columns

        ).sort_values(
            ascending=False
        )

        importance_path = (
            MODEL_DIR /
            "feature_importance.csv"
        )

        importance.to_csv(
            importance_path
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        importance.head(10).sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_title(
            "Top 10 Feature Importances - Optimized Random Forest"
        )

        ax.set_xlabel(
            "Importance"
        )

        fig.tight_layout()

        importance_plot_path = (
            MODEL_DIR /
            "feature_importance.png"
        )

        fig.savefig(
            importance_plot_path,
            dpi=220
        )

        plt.close(fig)

    # ------------------------------------------------------
    # Save final model
    # ------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "best_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nFinal model saved to:"
        f"\n{model_path}"
    )

    return metrics


# ==========================================================
# MLFLOW TRACKING
# ==========================================================

def log_mlflow(
    model,
    best_params,
    metrics
):

    print("\n" + "=" * 80)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 80)

    mlflow.set_tracking_uri(
        f"file:{MLRUNS_DIR.resolve()}"
    )

    mlflow.set_experiment(
        "Predictive_Maintenance_Project"
    )

    with mlflow.start_run(
        run_name="Final_Optimized_Random_Forest"
    ):

        # Model information
        mlflow.log_param(
            "Model",
            "Random Forest"
        )

        mlflow.log_param(
            "Tuning_Method",
            "GridSearchCV"
        )

        mlflow.log_param(
            "Cross_Validation_Folds",
            5
        )

        mlflow.log_param(
            "Scoring",
            "F1"
        )

        # Best hyperparameters
        for key, value in best_params.items():

            mlflow.log_param(
                key,
                value
            )

        # Evaluation metrics
        for key, value in metrics.items():

            mlflow.log_metric(
                key,
                float(value)
            )

        # Artifacts
        artifact_files = [

            MODEL_DIR /
            "baseline_model_comparison.csv",

            MODEL_DIR /
            "gridsearch_results.csv",

            MODEL_DIR /
            "best_parameters.json",

            MODEL_DIR /
            "final_model_metrics.csv",

            MODEL_DIR /
            "classification_report.csv",

            MODEL_DIR /
            "confusion_matrix.png",

            MODEL_DIR /
            "roc_curve.png",

            MODEL_DIR /
            "feature_importance.csv",

            MODEL_DIR /
            "feature_importance.png",
        ]

        for artifact in artifact_files:

            if artifact.exists():

                mlflow.log_artifact(
                    str(artifact)
                )

        # Log final model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Random_Forest_Model"
        )

    print(
        "✅ MLflow run completed successfully."
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 80)
    print(
        "PREDICTIVE MAINTENANCE - MODEL TRAINING PIPELINE"
    )
    print("=" * 80)

    # ------------------------------------------------------
    # 1. Load train/test from HF
    # ------------------------------------------------------

    train_df, test_df = (
        load_train_test_from_huggingface()
    )

    # ------------------------------------------------------
    # 2. Separate features and target
    # ------------------------------------------------------

    X_train = train_df.drop(
        columns=[TARGET]
    )

    y_train = train_df[TARGET]

    X_test = test_df.drop(
        columns=[TARGET]
    )

    y_test = test_df[TARGET]

    print(
        f"\nTraining features: {X_train.shape[1]}"
    )

    print(
        f"Training samples: {X_train.shape[0]}"
    )

    print(
        f"Testing samples: {X_test.shape[0]}"
    )

    # ------------------------------------------------------
    # 3. Baseline models
    # ------------------------------------------------------

    benchmark_models(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # ------------------------------------------------------
    # 4. Hyperparameter tuning
    # ------------------------------------------------------

    best_model, best_params = (
        tune_random_forest(
            X_train,
            y_train
        )
    )

    # ------------------------------------------------------
    # 5. Final evaluation
    # ------------------------------------------------------

    final_metrics = (
        final_model_evaluation(
            best_model,
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    # ------------------------------------------------------
    # 6. MLflow
    # ------------------------------------------------------

    log_mlflow(
        best_model,
        best_params,
        final_metrics
    )

    print("\n" + "=" * 80)
    print(
        "MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)


if __name__ == "__main__":

    main()
'''

train_path = Path(
    "predictive_maintenance_project/model_building/train.py"
)

train_path.write_text(
    train_code,
    encoding="utf-8"
)

print("✅ train.py created successfully:")
print(train_path)
