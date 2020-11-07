FROM tensorflow/tensorflow:latest-gpu-jupyter


###################################
#### Install Java 11
###################################
ARG JDK_VERSION=15.0.1
ARG TAR_JAVA_FILE=jdk_bin.tar.gz

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
