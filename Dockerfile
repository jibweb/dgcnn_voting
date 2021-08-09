FROM tensorflow/tensorflow:1.6.0-devel-gpu
MAINTAINER Jb Weibel (jb.weibel@gmail.com)

RUN apt-get update && apt-get install -y libpcl-dev \
                                         libproj-dev \
                                         python-tk \
                                         cmake
RUN pip install cython pyyaml progress
ENV PYTHONPATH /home/jbweibel/install/python:$PYTHONPATH
WORKDIR /home/jbweibel/code/gronet
