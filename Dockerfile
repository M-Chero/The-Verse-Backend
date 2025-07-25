FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
LABEL maintainer="Michelle-Cherotich <mkiprono@strathmore.edu>"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]