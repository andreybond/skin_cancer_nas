ARG TF_VERSION=2.1.0
FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3-jupyter
# FROM msranni/nni

RUN pip install autokeras

#FROM garawalid/autokeras:latest
##FROM tensorflow/tensorflow:1.13.1-gpu-py3
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install click
RUN pip install Sphinx
RUN pip install coverage
RUN pip install awscli
RUN pip install flake8
RUN pip install python-dotenv>=0.5.1
RUN pip install ediblepickle
RUN pip install hyperopt
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install pandas
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install jupyter

RUN pip install xlrd
RUN pip install imutils

RUN apt-get update

RUN ln -fs /usr/share/zoneinfo/Europe/Riga /etc/localtime
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y python-opencv

RUN pip install opencv-python
RUN pip install pillow

# Parquet implementation for storing pandas dataframes

#RUN pip install fastparquet 

# RUN pip install pyarrow  
# END of Parquet implementation
RUN pip install cmake
RUN pip install MulticoreTSNE
RUN pip install PyQt5

RUN pip install ipyparallel
RUN pip install sacred
RUN pip install pymongo
RUN apt install -y git
RUN export GIT_PYTHON_REFRESH=quiet

RUN apt-get update

# RUN pip install numpy==1.15.4
RUN pip install graphviz==0.8.4
RUN pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torch==1.0.0
# RUN pip install torchvision==0.2.1
RUN pip install tensorboard==1.13.0
RUN pip install tensorboardX==1.6

RUN pip install nni


# Just to recheck - let's install requirements which should be equal to the list above (if not, update list above)
# RUN pip install -r requirements.txt

# FROM manifoldai/orbyter-dl-dev:1.6

# RUN pip uninstall -y tensorflow
# RUN apt-get --purge remove "*cublas*" "cuda*"
# # RUN reboot
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
# RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# apt install cuda-10-0
# reboot 

# #RUN pip install -r requirements.txt
# #RUN rm -rf /build &&\
# #       rm -rf ~/.cache

# # FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# # # This part below should be same as in orbyter-base-sys/1.x/Dockerfile
# # # TODO is there a way to parameterize this?
# # COPY . /build
# # WORKDIR /build
# # ENV LC_ALL=C.UTF-8
# # ENV LANG=C.UTF-8
# # RUN apt-get update -qq && \
# #     apt-get install -qqy --no-install-recommends \
# #       software-properties-common &&\
# #       #python-software-properties && \
# # #
# #     export DEBIAN_FRONTEND=noninteractive &&\
# #     ln -fs /usr/share/zoneinfo/Europe/Riga /etc/localtime &&\
# #     apt-get install -y tzdata &&\
# #     dpkg-reconfigure --frontend noninteractive tzdata &&\ 
# # #    
# #     add-apt-repository -y ppa:jonathonf/python-3.6 && \
# #     apt-get update -qq && \
# #     apt-get install -qqy --no-install-recommends \
# #       vim \
# #       emacs \
# #       screen \
# #       git \
# #       wget \
# #       curl \
# #       build-essential \
# #       cmake \
# #       htop \
# #       python3.6 \
# #       python3.6-dev \
# #       python3.6-venv \
# #       python3.6-tk \
# #       python3-pip && \
# # #
# #     # apt-get install -y python3-notebook jupyter jupyter-core python-ipykernel &&\
# #     # pip3 install --upgrade pip \
# #     # pip install jupyter \
# # #
# #     rm -rf /var/lib/apt/lists/* && \
# #     rm -f /usr/bin/python3 && \
# #     rm -f /usr/bin/python && \
# #     ln -s /usr/bin/python3.6 /usr/bin/python3 && \
# #     ln -s /usr/bin/python3.6 /usr/bin/python && \
# #     ln -s /usr/bin/pip3 /usr/bin/pip && \
# #     rm -rf ~/.cache
# # WORKDIR /
# # RUN rm -rf /build
# # # 
# # # 
# # # PART 2
# # COPY . /build
# # WORKDIR /build
# # # Separely split pip to avoid pip install main issue
# # RUN pip install --upgrade pip
# # RUN apt-get update &&\
# #     apt-get install -y libpq-dev python3-dev &&\
# #     pip install wheel &&\
# #     pip install setuptools &&\
# #     pip install -r requirements.txt &&\
# #     jupyter contrib nbextension install --user
# # # START ffmpeg install specific lines
# # # We want the latest ffmpeg, but for that we need to add a ppa.
# # # But we have an error while adding a ppa. This is because of a bug in add-apt-repository python package at 
# # # /usr/bin/ (see https://
# # # stackoverflow.com/questions/42386097/python-add-apt-repository-importerror-no-module-named-apt-pkg)
# # # The following is a fix for this: use python3.5 to run. Without this we can't add new ppas
# # # This must be after pipenv installs above
# # #RUN sed -i '1 s/^.*$/\# \! \/usr\/bin\/python3.6/' /usr/bin/add-apt-repository &&\
# # #    /usr/bin/python3.6 /usr/bin/add-apt-repository -y ppa:jonathonf/ffmpeg-3  &&\
# # RUN apt-get update && \
# #     apt-get install -y ffmpeg
# # # END ffmpeg install specific lines
# # # Python installs
# # ENV PYTHONPATH="/mnt:${PYTHONPATH}"
# # WORKDIR /mnt
# # RUN rm -rf /build &&\
# #     rm -rf ~/.cache
