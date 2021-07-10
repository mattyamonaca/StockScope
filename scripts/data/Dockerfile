FROM python:3

COPY requirements.txt   ./

RUN pip install -r requirements.txt

COPY handler.py   ./
COPY s3_manager.py ./

CMD ["python3", "-m", "handler"]      