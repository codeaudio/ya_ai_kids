FROM python:3.10

RUN apt-get update -y && apt-get install rustc curl -y
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 -
ENV PATH=/opt/poetry/bin:$PATH
WORKDIR /app
COPY src .
RUN poetry config virtualenvs.create false && poetry install
CMD ["python", "main.py"]
