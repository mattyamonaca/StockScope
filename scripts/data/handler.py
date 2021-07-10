
import json
import urllib.parse
import boto3
import pandas as pd
from s3_manager import S3Manager

import investpy
from concurrent.futures import ThreadPoolExecutor

def handler(all=False):
    manager = S3Manager()

    print("process start")
    inputs = {}

    if all==False:
        df = get_stock_info("recent")
    else:
        inputs = {
            "from_date": "01/01/2020",
            "to_date": "31/12/2020"
            }
        df = get_stock_info("historical", inputs)  

    manager.save_parquet(
        df = df,
        bucket = "stock-scope-bucket",
        key = "stocks",
        filename = "us",
        cols = ["Date"]
    )
    print("process end")
    

def get_stock_info(method="recent", inputs={}):
    def get_recent_data(symbol):
        try :
            stock = investpy.get_stock_recent_data(stock=symbol,country=country)
        except:
            stock = None
        return stock
    
    def get_historical_data(symbol, from_date, to_date):
        try:
            stock=investpy.get_stock_historical_data(
                stock=symbol,
                country = country,
                from_date=from_date,
                to_date=to_date
            )
        except:
            stock=None
        return stock

    country = "united states"
    symbols = investpy.get_stocks_list(country)
    stocks = []

    if method == "recent":
        func = get_recent_data
        inputs_cp = inputs.copy()
    elif method == "historical":
        func = get_historical_data
        inputs_cp = inputs.copy()

    with ThreadPoolExecutor(max_workers=3) as executor:
        for symbol in symbols:
            inputs_cp["symbol"] = symbol
            future = executor.submit(func, **inputs_cp)
            stock = future.result()
            
            if stock is not None:
                stock["symbol"] = symbol
                stocks.append(stock)
            
    stock_info = pd.concat(stocks).reset_index()
    stock_info["Date"] = stock_info["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    return stock_info

if __name__ == "__main__":
    handler()
