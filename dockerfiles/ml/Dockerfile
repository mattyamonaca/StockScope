FROM python:3

COPY ./scripts/ml/requirements.txt   ./

RUN pip install -r requirements.txt

COPY ./scripts/ml/__init__.py ./
COPY ./scripts/ml/entry_point.py ./
COPY ./scripts/ml/config ./config
COPY ./scripts/ml/preprocess ./preprocess
COPY ./scripts/ml/model ./model

CMD ["python3", "-m", "entry_point"]