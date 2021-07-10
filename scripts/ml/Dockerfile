FROM python:3

COPY requirements.txt   ./

RUN pip install -r requirements.txt

COPY __init__.py ./
COPY entry_point.py ./
COPY config ./config
COPY preprocess ./preprocess
COPY model ./model

CMD ["python3", "-m", "entry_point"]