import pandas as pd
import yaml
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from .data_reader import pred_score


class Stacking:
    def __init__(self, model_list, start, end, ymd):
        self.start = start
        self.end = end
        self.ymd = ymd
        config_path = Path(__file__).parents[1].resolve() / "config" / "preprocess.yaml"
        with open(config_path) as f:
            config = yaml.load(f)

        self.df = self.data_read(model_list)
    
    def data_read(self, model_list :list) -> pd.DataFrame:
        df_list = []
        for name in model_list:
            path = Path(__file__).parents[1].resolve() / "result" / f"{name}_result.csv"
            tmp_df = pred_score(path)
            if len(df_list) == 0:
                df_list.append(tmp_df[["ymd", "code", "target"]])
            #tmp_df.columns = tmp_df.columns + "_" + name
            tmp_df[f"pred_{name}"] = tmp_df["pred"]

            df_list.append(tmp_df[["pred" + "_" + name, "ymd", "code"]])

        df = self.merge(df_list)
        return df

    def merge(self, df_list):
        base = df_list[0]
        for df in df_list[1:]:
            base = pd.merge(base, df, on = ["code", "ymd"], how = "left")
            print(len(base))
        return base
        
    
    def process(self):
        return self.df

    def data_split(self,df):

        print(self.ymd)
        
        train_df = df[df["ymd"] < self.ymd]
        test_df = df[df["ymd"] == self.ymd]

        
        X_train = train_df.drop(["target","code", "ymd"],axis=1)
        y_train = train_df["target"]
        X_test = test_df.drop(["target","code", "ymd"],axis=1)
        y_test = test_df[["ymd","code","target"]]

        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train)
        return X_train, X_val, X_test, y_train, y_val, y_test

#model = Stacking(["move_average", "ohlc"], "2020-11-30", "2021-01-27", "2021-01-20")
    
            
        
