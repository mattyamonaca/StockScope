import investpy
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def get_info(symbol):
    try :
        stock = investpy.get_stock_recent_data(stock=symbol,country=country)
    except:
        stock = None
    return stock


country = "united states"
symbols = investpy.get_stocks_list(country)
stocks = []
with ThreadPoolExecutor(max_workers=100) as executor:
    for symbol in symbols:
        future = executor.submit(get_info, symbol)
        if future is not None:
            stocks.append(future.result())
stock_info = pd.concat(stocks,axis=1)
print(stock_info)
