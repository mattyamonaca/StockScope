import pandas as pd
from chalice import Chalice
import awswrangler as wr
from datetime import datetime

app = Chalice(app_name="stockscope")

def get_latest_date(path_list, back):
    dates = []
    for path in path_list:
        dates.append(path.split("=")[1].replace("/",""))
    dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in dates], reverse=True)
    return dates[back]

def calc_date(key, back=0):
    if key == "stocks":
        base = "s3://stock-scope-bucket/stocks/us/"
    else:
        base = "s3://stock-scope-bucket/prediction/score/"    
    path_list = wr.s3.list_directories(base)
    date = get_latest_date([path.replace(base, "") for path in path_list], back)
    return date.strftime("%Y-%m-%d")


def read_s3(bucket, key, filename, filter = None):
    path = f"s3://{bucket}/{key}/{filename}"
    df = wr.s3.read_parquet(
        path = path,
        dataset = True,
        partition_filter = filter
        )
    return df

@app.route("/prediction/{method}", methods=["GET"], cors=True)
def prediction(method):
    ymd = calc_date(key="stocks")
    try:
        filter = lambda x:True if x["method"] == method and x["Date"] == ymd else False
        df = read_s3(
            bucket = "stock-scope-bucket",
            key = "prediction",
            filename = "score",
            filter = filter
        )
    
        df = df.sort_values("pred",ascending=False)
        df["pred"] = df["pred"].apply(lambda x: str(int(x*100)) + "%")
        return [x[1].to_dict() for x in df[:5].iterrows()]
    
    except Exception as e:
        response = {"error": str(e)}
    return response


@app.route("/prediction/explain/{method}", methods=["GET"], cors=True)
def model_explain(method):
    ymd = "2021-05-28" #calc_date(key="score")                                                                 
    drop_col = ["Date", "method"]
 
    try:
        filter = lambda x:True if x["method"] == method and x["Date"] == ymd else False
        df = read_s3(
            bucket = "stock-scope-bucket",
            key = "prediction",
            filename = "explain",
            filter = filter
        )
        res = []
        for key, item in df.to_dict().items():
            if key in drop_col:
                continue
            res.append({
                    "name":key, 
                    "importance": int(item[0])
                })
    except Exception as e:
        res = {"error": str(e)}
    return res


@app.route("/before_result/{user}", methods=["GET"], cors=True)
def before_result(user):
    ymd = calc_date(key="score")
    print(ymd)

    filter = lambda x:True if x["user"] == user and x["ymd"] == ymd else False
    df = read_s3(
        bucket = "stock-scope-bucket",
        key = "summary",
        filename = "performance",
        filter = filter
    )

    df = df.sort_values("ymd",ascending=False)
    return [x[1].to_dict() for x in df.iterrows()]



@app.route("/before_summary/{user}", methods=["GET"], cors=True)
def before_summary(user):
    start_ymd = calc_date(key="score")
    end_ymd = calc_date(key="score", back = 7)

    filter = lambda x:True if x["user"] == user and x["ymd"] <= start_ymd and x["ymd"] > end_ymd else False
    df = read_s3(
        bucket = "stock-scope-bucket",
        key = "summary",
        filename = "accuracy",
        filter = filter
    )

    df = df.sort_values("ymd",ascending=False)
    return [x[1].to_dict() for x in df.iterrows()]

@app.route("/current_summary/{user}", methods=["GET"], cors=True)
def current_summary(user):
    ymd = calc_date(key="score")

    filter = lambda x:True if x["user"] == user and x["ymd"] == ymd else False
    df = read_s3(
        bucket = "stock-scope-bucket",
        key = "summary",
        filename = "accuracy",
        filter = filter
    )

    df = df.sort_values("ymd",ascending=False)
    tmp = [x[1].to_dict() for x in df.iterrows()][0]
    response = [
        {"name" : "上昇率5%以上株数", "value" : tmp["correct"]},
        {"name" : "上昇率5%未満株数", "value" : tmp["all"] - tmp["correct"]}
    ]
    return response

@app.route("/prediction/metrics/{method}", methods=["GET"], cors=True)
def metrics(method):
    end_ymd = calc_date(key="score")
    start_ymd = calc_date(key="score",back=7) 

    filter = lambda x:True if x["method"] == method and end_ymd <= x["Date"] and x["Date"] >= start_ymd else False
    df = read_s3(
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "metrics",
        filter = filter
    )

    print(df)
    response = []
    for i in range(0,len(df)):
        tmp = {}
        [tmp.update({k:v[i]}) for k,v in df.to_dict().items()]
        response.append(tmp)
    return response

@app.route("/prediction/precision_recall/{method}", methods=["GET"], cors=True)
def precision_recall(method):
    ymd = "2021-06-24" #calc_date(key="score")

    filter = lambda x:True if x["method"] == method and ymd == x["Date"] else False
    df = read_s3(
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "precision_recall",
        filter = filter
    )

    length = len(df)
    idx_list = []
    print(length)
    for idx in range(0,length + 1):
        if idx % int((length/40)) == 0:
            idx_list.append(idx)
        elif idx == 0:
            idx_list.append(idx)
        elif idx > (length - 10):
            idx_list.append(idx - 2)
    df = df.iloc[idx_list].reset_index(drop=True)

    df["precision"] = df["precision"].apply(lambda x:int(x*100))
    df["recall"] = df["recall"].apply(lambda x:int(x*100))
    df["thresholds"] = df["thresholds"].apply(lambda x:int(x*100))
        
    response = []
    for i in range(0,len(df)):
        tmp = {}
        [tmp.update({k:v[i]}) for k,v in df.to_dict().items()]
        response.append(tmp)
    return response

before_result("default_user")
#print(precision_recall("bollinger_band"))

