import pandas as pd
import yaml
from pathlib import Path
from .data_reader import kabuka, meigara
from .util import read_s3, add_before_rate

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class BollingerBand:
    def __init__(self, ymd, mode="main"):
        self.ymd = ymd    
        if mode == "main":
            config_path = Path(__file__).parents[1].resolve() / "config" / "preprocess.yaml"
            with open(config_path) as f:
                self.config = yaml.load(f)


    def data_read(self, test_data=None):
        if test_data is None:
            self.df_dict = self.s3_read(self.config)
        else:
            self.df_dict = test_data
    
    def s3_read(self, config: dict) -> pd.DataFrame:
        df_dict = {}
        for k,v in config["PATH"].items():
            if k == "kabuka":
                df_dict["kabuka"] = read_s3(
                    bucket = v["bucket"],
                    key = v["key"],
                    filename = v["filename"]
                )
        return df_dict

    def data_build(self):
        self.df = self.df_dict["kabuka"]

    def process(self):
        df = self.df
    
        #移動平均
        df["day_9_mean"] = df["Close"].rolling(9).mean()
        df["day_10_mean"] = df["Close"].rolling(10).mean()
        df["day_20_mean"] = df["Close"].rolling(20).mean()

        #標準偏差                                                                                                                 
        df["day_9_std"] = df["Close"].rolling(9).std()
        df["day_10_std"] = df["Close"].rolling(10).std()
        df["day_20_std"] = df["Close"].rolling(20).std()
        

        #ボリンジャーバンド
        days = [9,10,20]
        sigmas = [1,2,3]
        for day in days:
            for sigma in sigmas:
                df[f"pos_{sigma}_{day}_band"] = df[f"day_{day}_mean"] + df[f"day_{day}_std"] * sigma
                df[f"neg_{sigma}_{day}_band"] = df[f"day_{day}_mean"] - df[f"day_{day}_std"] * sigma

                #乖離率
                """
                df[f"pos_{sigma}_{day}_dissociation"] = df["Close"]/df[f"pos_{sigma}_{day}_band"]
                df[f"neg_{sigma}_{day}_dissociation"] = df["Close"]/df[f"neg_{sigma}_{day}_band"]
                """

        df = self.encode(df)
        return df

    def encode(self, df):
        """
        category_cols = ["symbol"]
        for col in category_cols:
            target_col = df[col]
            le = preprocessing.LabelEncoder()
            le.fit(target_col)
            le_col = le.transform(target_col)
            df[col] = pd.Series(le_col).astype('category')
        """
        df["symbol"] = df["symbol"].astype("category")
        df["Date"] = df["Date"].astype("object")
        return df

    def data_split(self, df):
        df = add_before_rate(df)
        df["target"] = df["before_ratio"].apply(lambda x: 1 if float(x) > float(0.05) else 0)

        train_df = df[df["Date"] < self.ymd]
        test_df = df[df["Date"] == self.ymd]

        X_train = train_df.drop(["target", "before_ratio", "Date", "symbol", "Currency", "Volume"],axis=1)
        y_train = train_df["target"]

        X_test = test_df.drop(["target", "before_ratio", "Date", "symbol", "Currency", "Volume"],axis=1)
        y_test = test_df[["Date", "symbol", "target"]]

        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train)
        return X_train, X_val, X_test, y_train, y_val, y_test

    
            
        
