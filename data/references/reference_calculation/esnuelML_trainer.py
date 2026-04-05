#!/usr/bin/env python3

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn import metrics
from threadpoolctl import threadpool_limits


DEFAULT_SET_COLUMN = "Set"
DEFAULT_TRAIN_LABEL = "train"
DEFAULT_VAL_LABEL = "val"
DEFAULT_TEST_LABEL = "test"
STATIC_PARAMS = {
    "boosting": "gbdt",
    "objective": "regression",
    "metric": ["MAE", "RMSE"],
    "feature_pre_filter": False,
    "verbosity": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run inference with a LightGBM regressor for ESNUEL descriptors."
    )
    parser.add_argument(
        "--input-dataframe",
        required=True,
        help="Path to the input dataframe file (.csv, .csv.gz, .pkl, .pickle, .parquet).",
    )
    parser.add_argument(
        "--descriptor-column",
        required=True,
        help="Column name containing descriptor vectors.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target column name. Required for training, optional for prediction-only.",
    )
    parser.add_argument(
        "--num-cpu",
        type=int,
        default=1,
        help="Number of CPU cores used for Optuna and LightGBM.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where models, metrics, and prediction CSV files are written.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and only run prediction using --model-path.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a trained LightGBM model. Required when --predict-only is set.",
    )
    parser.add_argument(
        "--set-column",
        default=DEFAULT_SET_COLUMN,
        help=f"Column containing split labels. Default: {DEFAULT_SET_COLUMN}",
    )
    parser.add_argument(
        "--test-label",
        default=DEFAULT_TEST_LABEL,
        help=f"Label in --set-column used as held-out test split. Default: {DEFAULT_TEST_LABEL}",
    )
    parser.add_argument(
        "--train-label",
        default=DEFAULT_TRAIN_LABEL,
        help=f"Label in --set-column used for training. Default: {DEFAULT_TRAIN_LABEL}",
    )
    parser.add_argument(
        "--val-label",
        default=DEFAULT_VAL_LABEL,
        help=f"Label in --set-column used for validation. Default: {DEFAULT_VAL_LABEL}",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=500,
        help="Number of Optuna trials for hyperparameter optimization.",
    )
    parser.add_argument(
        "--optuna-num-boost-round",
        type=int,
        default=100,
        help="Maximum boosting rounds during Optuna CV.",
    )
    parser.add_argument(
        "--optuna-early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds during Optuna CV.",
    )
    parser.add_argument(
        "--train-num-boost-round",
        type=int,
        default=10000,
        help="Maximum boosting rounds when training fold models.",
    )
    parser.add_argument(
        "--train-early-stopping-rounds",
        type=int,
        default=250,
        help="Early stopping rounds when training fold models.",
    )

    args = parser.parse_args()
    if args.predict_only and not args.model_path:
        parser.error("--model-path is required when --predict-only is specified.")
    if not args.predict_only and not args.target_column:
        parser.error("--target-column is required unless --predict-only is specified.")
    if args.num_cpu < 1:
        parser.error("--num-cpu must be >= 1.")
    return args


def configure_threading(num_cpu: int) -> None:
    os.environ["MKL_NUM_THREADS"] = str(num_cpu)
    os.environ["MKL_DOMAIN_NUM_THREADS"] = f"MKL_BLAS={num_cpu}"
    os.environ["OMP_NUM_THREADS"] = "1"


def load_dataframe(path: str) -> pd.DataFrame:
    input_path = Path(path)
    suffixes = input_path.suffixes

    if suffixes[-2:] == [".csv", ".gz"] or suffixes[-1:] == [".csv"]:
        df = pd.read_csv(input_path)
    elif suffixes[-1:] in [[".pkl"], [".pickle"]]:
        df = pd.read_pickle(input_path)
    elif suffixes[-1:] == [".parquet"]:
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported dataframe format: {input_path}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def parse_descriptor_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Descriptor value is an empty string.")

        try:
            return np.asarray(json.loads(text), dtype=float)
        except json.JSONDecodeError:
            pass

        try:
            return np.asarray(ast.literal_eval(text), dtype=float)
        except (ValueError, SyntaxError):
            pass

        stripped = text.strip("[]()")
        if "," in stripped:
            parts = [part.strip() for part in stripped.split(",") if part.strip()]
        else:
            parts = [part for part in re.split(r"\s+", stripped) if part]

        if parts:
            try:
                return np.asarray([float(part) for part in parts], dtype=float)
            except ValueError as exc:
                raise ValueError(f"Failed to parse descriptor string: {text[:80]}") from exc

    raise TypeError(f"Unsupported descriptor value type: {type(value)!r}")


def prepare_dataframe(
    df: pd.DataFrame, descriptor_column: str, target_column: str
) -> pd.DataFrame:
    prepared = df.copy()
    prepared[descriptor_column] = prepared[descriptor_column].apply(parse_descriptor_value)
    if target_column:
        prepared[target_column] = prepared[target_column].astype(float)
    return prepared


def get_feature_matrix(df: pd.DataFrame, descriptor_column: str) -> np.ndarray:
    return np.asarray(df[descriptor_column].tolist(), dtype=float)


def build_objective(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    descriptor_column: str,
    target_column: str,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Any:
    def objective(trial: optuna.trial.Trial) -> float:
        print(f"Trial {trial.number}:")

        search_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 20, 60),
            "num_leaves": trial.suggest_int("num_leaves", 1024, 3072),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 1024),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        all_params = {**search_params, **STATIC_PARAMS}
        train_x = get_feature_matrix(df_train, descriptor_column)
        train_y = df_train[target_column].to_numpy(dtype=float)
        valid_x = get_feature_matrix(df_val, descriptor_column)
        valid_y = df_val[target_column].to_numpy(dtype=float)

        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)

        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, "rmse", valid_name="valid"
        )
        callbacks = [pruning_callback] if pruning_callback is not None else None

        print(all_params)
        model = lgb.train(
            all_params,
            dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
            verbose_eval=False,
        )

        preds = model.predict(valid_x, num_iteration=model.best_iteration)
        metrics_dict = evaluate_predictions(valid_y, preds)
        score = float(model.best_score["valid"]["rmse"])
        print(
            "Validation Score (RMSE): "
            f"{score:.4f}, Validation MSE: {metrics_dict['mse']:.4f}, "
            f"Validation RMSE: {metrics_dict['rmse']:.4f}, Validation MAE: {metrics_dict['mae']:.4f}, "
            f"Validation R2: {metrics_dict['r2']:.4f}"
        )
        print("-----------------------------------------------------------------\n")

        return score

    return objective


def save_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
    target_column: str = None,
) -> pd.DataFrame:
    prediction_df = df.copy()
    prediction_df["prediction"] = predictions
    if target_column and target_column in prediction_df.columns:
        prediction_df["residual"] = prediction_df[target_column] - prediction_df["prediction"]
    prediction_df.to_csv(output_path, index=False)
    return prediction_df


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    mse = metrics.mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "r2": float(metrics.r2_score(y_true, y_pred)),
    }


def train_model(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("----------------REGRESSOR SETUP----------------")
    print(f"Input dataframe:       {args.input_dataframe}")
    print(f"Descriptor column:     {args.descriptor_column}")
    print(f"Target column:         {args.target_column}")
    print(f"Number of CPUs:        {args.num_cpu}")
    print(f"Output directory:      {output_dir}")
    print()

    df = load_dataframe(args.input_dataframe)
    df = prepare_dataframe(df, args.descriptor_column, args.target_column)

    if args.set_column not in df.columns:
        raise KeyError(f"Column '{args.set_column}' was not found in the input dataframe.")

    split_labels = df[args.set_column].astype(str)
    df_train = df[split_labels == args.train_label].copy()
    df_val = df[split_labels == args.val_label].copy()
    df_test = df[split_labels == args.test_label].copy()
    if df_train.empty:
        raise ValueError("Training split is empty. Check --set-column and --train-label.")
    if df_val.empty:
        raise ValueError("Validation split is empty. Check --set-column and --val-label.")
    if df_test.empty:
        raise ValueError("Test split is empty. Check --set-column and --test-label.")

    x_test = get_feature_matrix(df_test, args.descriptor_column)
    y_test = df_test[args.target_column].to_numpy(dtype=float)

    print("----------------DATA SET----------------")
    print(
        f"Training size:         {df_train.shape[0]} "
        f"({df_train.shape[0] / df.shape[0]:.2f})"
    )
    print(
        f"Validation size:       {df_val.shape[0]} "
        f"({df_val.shape[0] / df.shape[0]:.2f})"
    )
    print(
        f"Held-out test size:    {df_test.shape[0]} "
        f"({df_test.shape[0] / df.shape[0]:.2f})"
    )
    print()

    print("----------------OPTUNA HYPERPARAMETER OPTIMIZATION----------------")
    print(f"Running {args.n_trials} rounds of LightGBM parameter optimisation:")

    objective = build_objective(
        df_train=df_train,
        df_val=df_val,
        descriptor_column=args.descriptor_column,
        target_column=args.target_column,
        num_boost_round=args.optuna_num_boost_round,
        early_stopping_rounds=args.optuna_early_stopping_rounds,
    )

    start = time.perf_counter()
    with threadpool_limits(limits=args.num_cpu, user_api="openmp"):
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            direction="minimize",
        )
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.num_cpu)
    finish = time.perf_counter()
    print(f"\nOptuna hyperparameter optimization finished in {round(finish - start, 2)} second(s)")

    study_path = output_dir / "optuna_study.pkl"
    joblib.dump(study, study_path)

    best_params = study.best_trial.params
    all_params = {**best_params, **STATIC_PARAMS}
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2)

    print(f"Best trial parameters:\n{json.dumps(all_params, indent=2)}")
    print()

    print("----------------TRAINING REGRESSOR----------------")
    train_x = get_feature_matrix(df_train, args.descriptor_column)
    train_y = df_train[args.target_column].to_numpy(dtype=float)
    valid_x = get_feature_matrix(df_val, args.descriptor_column)
    valid_y = df_val[args.target_column].to_numpy(dtype=float)

    train_dataset = lgb.Dataset(train_x, label=train_y)
    valid_dataset = lgb.Dataset(valid_x, label=valid_y, reference=train_dataset)

    with threadpool_limits(limits=args.num_cpu, user_api="openmp"):
        best_model = lgb.train(
            all_params,
            train_dataset,
            num_boost_round=args.train_num_boost_round,
            early_stopping_rounds=args.train_early_stopping_rounds,
            valid_sets=[train_dataset, valid_dataset],
            verbose_eval=200,
        )

    model_path = output_dir / "final_best_model.txt"
    best_model.save_model(str(model_path), num_iteration=best_model.best_iteration)
    val_preds = best_model.predict(valid_x, num_iteration=best_model.best_iteration)
    val_metrics = evaluate_predictions(valid_y, val_preds)
    save_predictions(df_val, val_preds, output_dir / "val_predictions.csv", args.target_column)
    with open(output_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    print()
    print("----------------TRAINING RESULTS----------------")
    print(f"Validation RMSE:       {val_metrics['rmse']:.4f}")
    print()

    final_model = lgb.Booster(model_file=str(model_path))
    y_preds = final_model.predict(x_test, num_iteration=final_model.best_iteration)

    prediction_path = output_dir / "test_predictions.csv"
    save_predictions(df_test, y_preds, prediction_path, args.target_column)

    test_metrics = evaluate_predictions(y_test, y_preds)
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("----------------TEST RESULTS----------------")
    print(f"Pred. MSE:  {test_metrics['mse']:.6f}")
    print(f"Pred. RMSE: {test_metrics['rmse']:.6f}")
    print(f"Pred. MAE:  {test_metrics['mae']:.6f}")
    print(f"Pred. R2:   {test_metrics['r2']:.6f}")
    print()
    print(f"Saved model to:        {model_path}")
    print(f"Saved predictions to:  {prediction_path}")
    print(f"Saved metrics to:      {output_dir / 'test_metrics.json'}")


def predict_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.input_dataframe)
    df = prepare_dataframe(df, args.descriptor_column, args.target_column)
    features = get_feature_matrix(df, args.descriptor_column)

    model = lgb.Booster(model_file=args.model_path)
    predictions = model.predict(features, num_iteration=model.best_iteration)

    prediction_path = output_dir / "predictions.csv"
    save_predictions(df, predictions, prediction_path, args.target_column)

    print("----------------PREDICTION RESULTS----------------")
    print(f"Saved predictions to:  {prediction_path}")

    if args.target_column and args.target_column in df.columns:
        y_true = df[args.target_column].to_numpy(dtype=float)
        metrics_dict = evaluate_predictions(y_true, predictions)
        metrics_path = output_dir / "prediction_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Saved metrics to:      {metrics_path}")
        print(f"Pred. RMSE:            {metrics_dict['rmse']:.6f}")


def main() -> None:
    args = parse_args()
    configure_threading(args.num_cpu)

    if args.predict_only:
        predict_only(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
