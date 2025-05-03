import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def split_stratified_into_train_val_test(
    df_input,
    stratify_colname="y",
    frac_train=0.6,
    frac_val=0.15,
    frac_test=0.25,
    random_state=None,
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    """

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(
            "fractions %f, %f, %f do not add up to 1.0"
            % (frac_train, frac_val, frac_test)
        )

    if stratify_colname not in df_input.columns:
        raise ValueError("%s is not a column in the dataframe" % (stratify_colname))

    X = df_input  # Contains all columns.
    y = df_input[
        [stratify_colname]
    ]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_state
    )

    if frac_val <= 0:
        assert len(df_input) == len(df_train) + len(df_temp)
        return df_train, pd.DataFrame(), df_temp, y_train, pd.DataFrame(), y_temp

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        stratify=y_temp,
        test_size=relative_frac_test,
        random_state=random_state,
    )

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test, y_train, y_val, y_test


def run_classification(
    model: Pipeline,
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: DataFrame,
    y_test: DataFrame,
) -> Dict:
    result = {}
    y_train_predict = model.predict(X_train)
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_test_predict = np.where(y_test_probs > 0.5, 1, 0)

    result["pipeline"] = model
    result["probs"] = y_test_probs
    result["preds"] = y_test_predict

    result["Precision_train"] = metrics.precision_score(y_train, y_train_predict)
    result["Precision_test"] = metrics.precision_score(y_test, y_test_predict)
    result["Recall_train"] = metrics.recall_score(y_train, y_train_predict)
    result["Recall_test"] = metrics.recall_score(y_test, y_test_predict)
    result["Accuracy_train"] = metrics.accuracy_score(y_train, y_train_predict)
    result["Accuracy_test"] = metrics.accuracy_score(y_test, y_test_predict)
    result["ROC_AUC_test"] = metrics.roc_auc_score(y_test, y_test_probs)
    result["F1_train"] = metrics.f1_score(y_train, y_train_predict)
    result["F1_test"] = metrics.f1_score(y_test, y_test_predict)
    result["MCC_test"] = metrics.matthews_corrcoef(y_test, y_test_predict)
    result["Cohen_kappa_test"] = metrics.cohen_kappa_score(y_test, y_test_predict)
    result["Confusion_matrix"] = metrics.confusion_matrix(y_test, y_test_predict)

    return result


def run_regression(
    model: Pipeline,
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: DataFrame,
    y_test: DataFrame,
) -> Dict:
    result = {}
    y_train_pred = model.predict(X_train.values)
    y_test_pred = model.predict(X_test.values)

    result["fitted"] = model
    result["train_preds"] = y_train_pred
    result["preds"] = y_test_pred

    result["RMSE_train"] = math.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    result["RMSE_test"] = math.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    result["RMAE_test"] = math.sqrt(metrics.mean_absolute_error(y_test, y_test_pred))
    result["R2_test"] = metrics.r2_score(y_test, y_test_pred)

    return result
