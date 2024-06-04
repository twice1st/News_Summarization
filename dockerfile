FROM python:3.12

WORKDIR /app

COPY ./summarization.py ./summarization.py
COPY ./requirements.txt ./requirements.txt
COPY ./main.py ./main.py

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 3000

CMD ["uvicorn", "main:app"]