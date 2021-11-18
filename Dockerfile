FROM python:3.7-slim-buster

ENV APP_DIRECTORY=/app

WORKDIR $APP_DIRECTORY

# Framework Dependencies
RUN apt-get update && apt-get install -y \
      libsnappy-dev \
      vim \
      libgomp1

COPY poetry.lock pyproject.toml main.py $APP_DIRECTORY/

#COPY data $APP_DIRECTORY/data

RUN mkdir $APP_DIRECTORY/secrets && \
    touch $APP_DIRECTORY/secrets/.env  && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction

COPY ./finance_ml $APP_DIRECTORY/finance_ml

EXPOSE 22222

CMD ["ddtrace-run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "22222", "--no-access-log"]
