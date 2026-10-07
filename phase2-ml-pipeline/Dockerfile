FROM apache/airflow:2.7.3-python3.10

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /opt/airflow

COPY requirements.txt ./

USER airflow
RUN pip install --no-cache-dir -r requirements.txt
