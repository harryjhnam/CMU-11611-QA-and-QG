# Ubuntu Linux as the base image. You can use any version of Ubuntu here
FROM ubuntu:18.04
# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# Install Python
RUN apt-get -y update && \
apt-get -y upgrade
# The following line ensures that the subsequent install doesn't expect user input
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3-pip python3-dev
# Upgrade pip
RUN pip3 install --upgrade pip

# Add data into container
COPY data/ /host/Users/data

# Add the files into container, under QA folder
COPY QA/ /QA

# Add the files into container, under QG folder
COPY QG/ /QG

# Add programs into container
ADD ask /
ADD answer /

# install requirements
RUN pip3 install -r QA/requirement.txt
RUN pip3 install -r QG/requirement.txt

# Change the permissions of programs and download pretrained models
CMD ["chmod 777 /*"]
RUN python3 /QA/download_pretrained.py
RUN python3 /QG/download_pretrained.py
RUN python3 -m spacy download en_core_web_sm

# Set working dir as /
WORKDIR /
ENTRYPOINT ["/bin/bash", "-c"]