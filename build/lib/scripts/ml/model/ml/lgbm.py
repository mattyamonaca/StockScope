
import pandas as pd
from pathlib import Path
import yaml

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

#from sklearn.metrics import mean_squared_error 

class LgbModel:
    def __init__(self, name, X_train, y_train, X_val, y_val):
        self.name = name
        config_path = Path(__file__).parents[2].resolve() / "config" / "param.yaml"
        with open(config_path) as f:
            self.params = yaml.load(f)
        self.headers = X_train.columns
        self.lgb_train = lgb.Dataset(X_train,y_train)
        self.lgb_eval = lgb.Dataset(X_val,y_val,reference = self.lgb_train)
        self.model = None
            
    def train(self):
        params = self.params[self.name]["lgbm_param"]
        print(params)
        
        model = lgb.train(
            params,
            self.lgb_train,
            valid_sets = self.lgb_eval,
            verbose_eval = 10,
            num_boost_round = 1000,
            early_stopping_rounds = 100
        )
        
        self.model = model

    def predict(self,X_test):
        return self.model.predict(X_test, num_iteration = self.model.best_iteration)
        

    def validation(self, y_test, y_pred):
        print(pd.concat([y_test["target"].reset_index(), pd.DataFrame(y_pred)],axis=1))
        if self.params[self.name]["lgbm_param"]["objective"] == "regression":
            mse = metrics.mean_squared_error(y_test["target"], y_pred)
            auc = np.sqrt(mse)
        if self.params[self.name]["lgbm_param"]["objective"] == "binary":
            fpr, tpr, thresholds = metrics.roc_curve(y_test["target"],y_pred)                                                     
            auc = metrics.auc(fpr,tpr)
            precision, recall, pr_thresholds = metrics.precision_recall_curve(y_test["target"],y_pred)
            
        precision_recall = pd.DataFrame([precision,recall,pr_thresholds]).T
        precision_recall.columns = ["precision","recall","thresholds"]

        pred = pd.DataFrame(y_pred)
        pred.columns = ["pred"]
        result = pd.concat([y_test.reset_index(drop=True),pred],axis=1)

        max_prs = precision_recall[precision_recall["precision"].max() == precision_recall["precision"]]
        max_th = max_prs["thresholds"].min()
        max_pr = max_prs["precision"].max()
        
        metric = pd.DataFrame({
            "auc": [float(auc)],
            "thresholds": [float(max_th)],
            "precision": [float(max_pr)]
            })
        return result, precision_recall, metric

    def get_importance(self):
        importance = pd.DataFrame(self.model.feature_importance(), index = self.headers, columns=["importance"])
        return importance
            
