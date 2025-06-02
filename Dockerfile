FROM python:3.12 AS build

WORKDIR /app
COPY . /app
RUN make


FROM python:3.12-alpine

WORKDIR /app
COPY --from=build /app /app
CMD ["python", "trainer.py"]