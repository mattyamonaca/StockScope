import pandas as pd
import yaml
from pathlib import Path
from .data_reader import kabuka, meigara, correlation

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class Correlation:
    def __init__(self, ymd):
        self.ymd = ymd
        config_path = Path(__file__).parents[1].resolve() / "config" / "preprocess.yaml"
        with open(config_path) as f:
            config = yaml.load(f)
        path_dict = {
            "kabuka" : config["PATH"]["kabuka"],
            "meigara" : config["PATH"]["meigara"],
            "correlation" : config["PATH"]["correlation"]
        }
        self.df_dict = self.data_read(path_dict)
    
    def data_read(self, path_dict :dict) -> pd.DataFrame:
        df_dict = {}
        for k,v in path_dict.items():
            path = Path(__file__).parents[1].resolve() / path_dict[k]
            if k == "kabuka":
                df_dict["kabuka"] = kabuka(path)
            elif k == "meigara":
                df_dict["meigara"] = meigara(path)
            elif k == "correlation":
                df_dict["correlation"] = correlation(path)
                
        return df_dict

    def merge(self):
        self.df = pd.merge(self.df_dict["kabuka"], self.df_dict["meigara"], on="code", how="left")
        self.df = pd.merge(self.df, self.df_dict["correlation"], on = ["code", "ymd"], how="left")

    def process(self):
        self.merge()
        df = self.encode(self.df)
        return df

    def encode(self, df):
        category_cols = ["name","market"]
        for col in category_cols:
            target_col = df[col]
            le = preprocessing.LabelEncoder()
            le.fit(target_col)
            le_col = le.transform(target_col)
            df[col] = pd.Series(le_col).astype('category')

        df["code"] = df["code"].astype("category")
        return df

    def data_split(self,df):
        df["target"] = df["before_ratio"].apply(lambda x: 1 if float(x) > float(0.05) else 0)

        train_df = df[df["ymd"] < self.ymd]
        test_df = df[df["ymd"] == self.ymd]
        X_train = train_df.drop(["target","before_ratio","ymd","Unnamed: 0","name"],axis=1)
        y_train = train_df["target"]
        X_test = test_df.drop(["target","before_ratio","ymd","Unnamed: 0","name"],axis=1)
        y_test = test_df[["ymd","code","before_ratio","target"]]

        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train)
        return X_train, X_val, X_test, y_train, y_val, y_test

    
            
        
