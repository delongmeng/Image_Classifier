FROM python:3.8
COPY requirements.txt /classifier/requirements.txt
WORKDIR /classifier
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /classifier
COPY . .
CMD ["python", "predict.py"]
