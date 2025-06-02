FROM python:3.12 AS build

WORKDIR /app
COPY . /app
RUN make

CMD ["sh", "-c", ". venv/bin/activate; python trainer.py"]