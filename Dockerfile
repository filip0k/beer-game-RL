# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /beer-game

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY beer-game-environment-0.1.tar.gz beer-game-environment-0.1.tar.gz

RUN pip3 install ./beer-game-environment-0.1.tar.gz

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . /beer-game/

#CMD ["python", "./one_trained_policy.py"]
