import pandas as pd
from pathlib import Path
import yaml

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

class LinnerModel:
    def __init__(self, name):
        self.name = name
        config_path = Path(__file__).parents[2].resolve() / "config" / "param.yaml"
        with open(config_path) as f:
            self.params = yaml.load(f)

        self.model = None
            
    def train(self,X_train,y_train,X_val,y_val):
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        
        self.model = lr
        
    def predict(self,X_test):
        return self.model.predict_proba(X_test)[:,1]
        
    def validation(self, X_train, y_test, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_test["target"],y_pred)
        auc = metrics.auc(fpr,tpr)
        #importance = pd.DataFrame(self.model.feature_importance(), index = X_train.columns,columns=["importance"])
        pred = pd.DataFrame(y_pred)
        pred.columns = ["pred"]
        result = pd.concat([y_test.reset_index(drop=True),pred],axis=1)
        
        return result, auc
            
