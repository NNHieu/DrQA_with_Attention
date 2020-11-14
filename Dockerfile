FROM tensorflow/tensorflow:latest-gpu-jupyter


###################################
#### Install Java 11
###################################
ARG JDK_VERSION=11.0.2
ARG TAR_JAVA_FILE=openjdk-11.0.2_linux-x64_bin.tar.gz

COPY ${TAR_JAVA_FILE} /tmp
ENV JAVA_HOME /usr/lib/jvm/jdk-${JDK_VERSION}
ARG INSTALL_DIR=/usr/bin
WORKDIR $INSTALL_DIR
RUN mkdir /usr/lib/jvm && tar xvf /tmp/${TAR_JAVA_FILE} -C /usr/lib/jvm \
    && ln -s $JAVA_HOME ${INSTALL_DIR}/java \
    && rm -rf $JAVA_HOME/man /tmp/${TAR_JAVA_FILE}
ENV PATH $PATH:$JAVA_HOME/bin

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt

###################################
#### Transformer
###################################
ARG TRANSFORMERS_MASTER=transformers-master.zip

COPY ${TRANSFORMERS_MASTER} /tmp
RUN unzip /tmp/${TRANSFORMERS_MASTER} -d /tmp && cd /tmp/transformers-master && pip install --upgrade . && rm -r /tmp/transformers-master /tmp/${TRANSFORMERS_MASTER}
WORKDIR /code
