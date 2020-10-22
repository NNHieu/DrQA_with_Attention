FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
