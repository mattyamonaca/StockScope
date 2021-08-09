from setuptools import setup, find_packages

setup(
    name="stockscope",
    version="0.0.1",
    packages=[
        "scripts", 
        "scripts/ml", 
        "scripts/ml/preprocess", 
        "scripts/ml/config", 
        "scripts/ml/util", 
        "scripts/ml/model",
        "scripts/ml/model/ml"
    ]
)
