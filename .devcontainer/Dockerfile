FROM webera/base

# Install required packages and Python 3.12
RUN apt update && \
    apt install -y \
    software-properties-common \
    gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-distutils \
    wget
    
# Set Python 3.12 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 && \
    update-alternatives --set python3 /usr/bin/python3.12

# Download and install the latest version of pip and setuptools
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Install Poetry
RUN pip install poetry

# Change the working directory
WORKDIR /workspaces/openvela