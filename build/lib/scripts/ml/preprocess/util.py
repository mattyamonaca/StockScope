import pandas as pd
#import yaml
#from pathlib import Path
#from .data_reader import kabuka, meigara

#from sklearn.model_selection import train_test_split
#from sklearn import preprocessing

import pandas as pd
import awswrangler as wr

def check_ymd(ymd):
        config_path = Path(__file__).parents[1].resolve() / "config" / "preprocess.yaml"
        with open(config_path) as f:
            config = yaml.load(f)
        path_dict = {
            "kabuka" : config["PATH"]["kabuka"]
        }
        path = Path(__file__).parents[1].resolve() / path_dict["kabuka"]
        df = kabuka(path)
       
        return ymd & set(df["ymd"])

def add_before_rate(df: pd.DataFrame):
    def calc_before_rate(df):
        df["day_rate"] = ((df["Close"]/df["Open"]) - 1)
        df["before_ratio"] = df["day_rate"].shift(-1)
        df = df.sort_values("Date")
        return df["before_ratio"]
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)
    pds = df.groupby("symbol").apply(lambda x: calc_before_rate(x)).reset_index()["before_ratio"]
    df["before_ratio"] = pds
    return df

def read_s3(bucket, key, filename):
	path = f"s3://{bucket}/{key}/{filename}"
	df = wr.s3.read_parquet(
		path = path,
		dataset = True
		)
	return df

def save_parquet(df, bucket, key, filename, cols=[]):
    path = f"s3://{bucket}/{key}/{filename}"
    wr.s3.to_parquet(
        df = df,
        path = path,
        partition_cols = cols,
        dataset = True,
        mode = "overwrite_partitions"
    )


if __name__ == "__main__":
   """	
   pdf = {
		"Close":[10,20, 30 , 50, 40],
		"Open":[5,20, 20 , 50, 30],
		"symbol":["aa","bb", "aa" , "bb", "bb"],
		"Date": ["2021-01-01","2021-01-01","2021-01-02","2021-01-03","2021-01-02",]
		}
   add_before_rate(pd.DataFrame(pdf))
   """
   df = read_s3("stock-scope-bucket", "stocks", "us")
   df.groupby("symbol").count().to_csv("survey.csv")
