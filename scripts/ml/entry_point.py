from scripts.ml.preprocess.processor import Processor
from scripts.ml.preprocess.util import read_s3
from scripts.ml.preprocess.move_average import MoveAverage
from scripts.ml.preprocess.ohlc import Ohlc
from scripts.ml.preprocess.bollinger_band import BollingerBand
from scripts.ml.preprocess.stochastics import Stochastics
from scripts.ml.preprocess.correlation import Correlation
from scripts.ml.model.ml.lgbm import LgbModel
from scripts.ml.model.ml.linner import LinnerModel
from scripts.ml.preprocess.stacking import Stacking
from scripts.ml.preprocess.util import check_ymd, save_parquet
import glob

import pandas as pd
import gc

import argparse

import os


def data_make(process_name, ymd):
    if process_name == "move_average":
        processor = Processor(MoveAverage, ymd)
    elif process_name == "ohlc":
        processor = Processor(Ohlc, ymd)
    elif process_name == "bollinger_band":
        processor = Processor(BollingerBand, ymd)
    elif process_name == "stochastics":
        processor = Processor(Stochastics, ymd)
    elif process_name == "correlation":
        processor = Processor(Correlation, ymd)
    return processor.run()

def train(name, ymd):
    X_train, X_val, X_test, y_train, y_val, y_test = data_make(name, ymd)
 
    model = LgbModel(name, X_train, y_train, X_val, y_val)
    model.train()
    y_pred = model.predict(X_test)

    score, precision_recall, metrics = model.validation(y_test, y_pred) 
    explain = pd.DataFrame(model.get_importance(), index=X_train.columns).T
    
    for data in [score, precision_recall, metrics, explain]:
        data["method"] = name
        data["Date"] = ymd
    
    save_score(score)
    save_metrics(metrics)
    save_explain(explain)
    save_precision_recall(precision_recall)

def save_score(score):
    save_parquet(
        df = score,
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "score",
        cols = ["Date", "method"]
    )

def save_metrics(metrics):
    save_parquet(
        df = metrics,
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "metrics",
        cols = ["Date", "method"]
    )

def save_precision_recall(precision_recall):
    save_parquet(
        df = precision_recall,
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "precision_recall",
        cols = ["Date", "method"]
    )

def save_explain(explain):
    save_parquet(
        df = explain,
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "explain",
        cols = ["Date", "method"]
    )


def ensemble(model_list, start, end, ymd):
    for model in model_list:
        files = glob.glob("./result/*")
        simulate(model, start, end)

    date_idx = set([ymd.strftime("%Y-%m-%d") for ymd in pd.date_range(ymd,end)])
    date_idx = check_ymd(date_idx)

    result_list = []
    for ymd in date_idx:
        processor = Stacking(model_list, start, end, ymd)
        df = processor.process()
        print(df)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.data_split(df)
        model = LinnerModel(model_list)
        model.train(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_test)
        result, auc = model.validation(y_test, y_pred)
        print(auc)
        result_list.append(result)
        gc.collect()

    results = pd.concat(result_list)
    results.sort_values("pred").to_csv("./result/stacking_result.csv", index=False)

    
def simulate(name, start, end):
    result_list = []
    score_df = pd.read_csv(f"./result/{name}_result.csv")

    ymd_list = set(score_df["ymd"])
    date_idx = set([ymd.strftime("%Y-%m-%d") for ymd in pd.date_range(start,end)])
    date_idx = date_idx - ymd_list
    date_idx = check_ymd(date_idx)

    if len(date_idx) == 0:
        print(f"{name} is all updated")
        return
    
    for ymd in date_idx:
        print(f"{name}_{ymd}_start")
        result = train(name, ymd)
        result_list.append(result)

    results = pd.concat(result_list)
    results = pd.concat([score_df, results])
    results.sort_values("pred").to_csv(f"./result/{name}_result.csv", index=False)
    
    
    print(results)
    

if __name__ == "__main__":
    print("learning start")
    alg =  os.environ["ALG"]
    ymd =  os.environ["YMD"]
    result = train(alg, ymd)

    print("learning end")
    
    
