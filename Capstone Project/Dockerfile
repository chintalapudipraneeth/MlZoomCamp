FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv lock
RUN pipenv install --system --deploy

COPY ["predict.py",  "./"]
COPY ["models/XgboostClassifier.bin", "./models/XgboostClassifier.bin"]

EXPOSE 8080

ENTRYPOINT ["waitress-serve"]
CMD ["--port=8080", "predict:app"]