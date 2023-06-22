FROM python:3

COPY ./scripts/ml/requirements.txt   ./
COPY ./scripts ./scripts
COPY ./setup.py ./setup.py

RUN pip install .
RUN pip install -r requirements.txt
WORKDIR /scripts/ml

CMD ["python3", "-m", "entry_point"]