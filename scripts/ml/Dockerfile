FROM python:3

COPY requirements.txt   ./

RUN pip install -r requirements.txt
RUN pip list

COPY __init__.py ./
COPY entry_point.py ./
COPY config ./config
COPY preprocess ./preprocess
COPY model ./model

WORKDIR /scripts/ml

RUN ["python3", "-m", "entry_point"]