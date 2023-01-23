FROM python:3.8-slim-buster

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

RUN pip install pillow gunicorn flask

COPY ["train.py", "predict.py", "CS107_0.995.h5", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]