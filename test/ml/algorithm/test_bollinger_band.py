import pytest
from scripts.ml.entry_point import run

def test_bollinger_band():
    alg = "bollinger_band"
    ymd = "2021-07-01"
    run(alg, ymd, mode="test")