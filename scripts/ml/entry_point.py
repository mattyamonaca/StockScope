from ml.preprocess.processor import Processor
from ml.preprocess.util import read_s3
from ml.preprocess.move_average import MoveAverage
from ml.preprocess.ohlc import Ohlc
from ml.preprocess.bollinger_band import BollingerBand
from ml.preprocess.stochastics import Stochastics
from ml.preprocess.correlation import Correlation
from ml.model.ml.lgbm import LgbModel
from ml.model.ml.linner import LinnerModel
from ml.preprocess.stacking import Stacking
from ml.preprocess.util import check_ymd, save_parquet
from ml.util import writer
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

def train(name, ymd, X_train, y_train, X_val, y_val):
    model = LgbModel(name, X_train, y_train, X_val, y_val)
    model.train()
    return model

def predict(name, ymd, model, X_test, y_test):
    y_pred = model.predict(X_test)

    score, precision_recall, metrics = model.validation(y_test, y_pred) 
    explain = pd.DataFrame(model.get_importance(), index=X_test.columns).T
    
    for data in [score, precision_recall, metrics, explain]:
        data["method"] = name
        data["Date"] = ymd
    
    return score, metrics, explain, precision_recall

def run(name, ymd, mode = "test"):
    X_train, X_val, X_test, y_train, y_val, y_test = data_make(name, ymd)
    model = train(name, ymd, X_train, y_train, X_val, y_val)
    score, metrics, explain, precision_recall = predict(name, ymd, model, X_test, y_test)

    if mode == "main":
        writer.write("score", score)   
        writer.write("metrics", metrics)
        writer.write("explain", explain)
        writer.write("precision_recall", precision_recall)
    

if __name__ == "__main__":
    print("learning start")
    alg =  os.environ["ALG"]
    ymd =  os.environ["YMD"]
    run(alg, ymd, mode="main")
    print("learning end")
    
    
