
import pandas as pd
import glob
import os

def kabuka(path):  
    files = glob.glob(os.path.join(path,"*.csv"))
    df_list = []
    
    for file in files:
        tmp_df = pd.read_csv(file)
        tmp_df["code"] = os.path.basename(file).split(".")[0]
        tmp_df["day_rate"] = ((tmp_df["end"]/tmp_df["start"]) - 1)
        tmp_df["before_ratio"] = tmp_df["day_rate"].shift(-1)
        #tmp_df = tmp_df.iloc[::-1]

        #print(tmp_df)
        df_list.append(tmp_df)

    df = pd.concat(df_list,ignore_index=True)

    return df

def meigara(path):
    files = glob.glob(os.path.join(path,"*.csv"))
    df_list = []

    for file in files:
        tmp_df = pd.read_csv(file)
        df_list.append(tmp_df)

    df = pd.concat(df_list,ignore_index=True)
    df["code"] = df["code"].astype(str)

    return df

def pred_score(path):
    df = pd.read_csv(path)
    df["code"] = df["code"].astype(str)
    df["ymd"] = df["ymd"].astype(str)
    df = df.drop_duplicates(["ymd","code"])
    return df

def correlation(path):
    df = pd.read_csv(path)
    df["code"] = df["code"].astype(str)
    return df
