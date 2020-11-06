FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt

###################################
#### Install Java 11
###################################
COPY openjdk-11.0.2_linux-x64_bin.tar.gz /tmp
ARG TAR_JAVA_FILE=${TAR_JAVA_FILE:-openjdk-11.0.2_linux-x64_bin.tar.gz}
ENV JAVA_HOME /usr/lib/jvm/jdk-11.0.2
ENV INSTALL_DIR /usr/bin
WORKDIR $INSTALL_DIR
RUN mkdir /usr/lib/jvm && tar xvf /tmp/${TAR_JAVA_FILE} -C /usr/lib/jvm \
    && ln -s $JAVA_HOME $INSTALL_DIR/java \
    && rm -rf $JAVA_HOME/man /tmp/${TAR_JAVA_FILE}
ENV PATH $PATH:$JAVA_HOME/bin
WORKDIR /code