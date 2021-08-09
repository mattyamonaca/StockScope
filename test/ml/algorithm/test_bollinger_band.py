import pytest
import pandas as pd
from scripts.ml.preprocess.bollinger_band import BollingerBand
from pandas.util.testing import assert_frame_equal

def test_bollinger_band():
    
    kabuka = pd.DataFrame(
        {
            "Date": [
                "2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14",
                "2021-06-15","2021-06-16","2021-06-17","2021-06-18","2021-06-19",
                "2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24",
                "2021-06-25","2021-06-26","2021-06-27","2021-06-28","2021-06-29",
            ],
            "Close": [
                100,110,120,130,140,
                150,140,130,120,110,
                100,110,120,130,140,
                150,140,130,120,110,
            ],
            "symbol": [
                "AAA","AAA","AAA","AAA","AAA",
                "AAA","AAA","AAA","AAA","AAA",
                "AAA","AAA","AAA","AAA","AAA",
                "AAA","AAA","AAA","AAA","AAA",
            ]
        }
    )

    ans = pd.DataFrame(
        {
            "Date": [
                "2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14",
                "2021-06-15","2021-06-16","2021-06-17","2021-06-18","2021-06-19",
                "2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24",
                "2021-06-25","2021-06-26","2021-06-27","2021-06-28","2021-06-29",
            ],
            "Close": [
                100,110,120,130,140,
                150,140,130,120,110,
                100,110,120,130,140,
                150,140,130,120,110,
            ]
        }
    )

    test_data = {
        "kabuka": kabuka
        }

    preprocessor = BollingerBand(ymd="2021-07-01", mode ="test")
    preprocessor.data_read(test_data=test_data)
    preprocessor.data_build()
    df = preprocessor.process()
    result = df[["Date","Close"]]
    print(ans)
    print(result)
    assert_frame_equal(result, ans)

test_bollinger_band()