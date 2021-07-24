from scripts.ml.preprocess.util import check_ymd, save_parquet

def write(name, data):
    if name not in globals().keys():
        raise ValueError(f"Error {name} is not definition")
    else:
        globals()[name](data)

def score(score):
    save_parquet(
        df = score,
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "score",
        cols = ["Date", "method"]
    )

def metrics(metrics):
    save_parquet(
        df = metrics,
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "metrics",
        cols = ["Date", "method"]
    )

def precision_recall(precision_recall):
    save_parquet(
        df = precision_recall,
        bucket = "stock-scope-bucket",
        key = "prediction",
        filename = "precision_recall",
        cols = ["Date", "method"]
    )
