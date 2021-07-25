FROM python:3

COPY ./scripts/ml/requirements.txt   ./
COPY ./scripts ./scripts
COPY ./setup.py ./setup.py

RUN python setup.py build
RUN python setup.py install

RUN pip install -r requirements.txt

CMD ["python3", "-m", "./scripts/ml/entry_point"]