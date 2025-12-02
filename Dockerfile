FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ENV CUDA_HOME=/usr/local/cuda-11.1/
# requirements
COPY .condarc /root
RUN conda init bash && \
conda clean -i && \
conda update --name base --channel defaults conda --yes && \
conda clean --all --yes  && \
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


RUN pip install mmcv-full==1.4.4 mmsegmentation==0.22.1  
RUN pip install tqdm thop tensorboard ipdb h5py ipython Pillow==9.5.0
RUN pip install -U numpy 


WORKDIR /completion_former 
COPY . /completion_former

# RUN cd src/model/deformconv/ && python setup.py build install
# nvidia apex
# RUN apt update --allow-unauthenticated --allow-insecure-repositories && \
#     DEBIAN_FRONTEND=noninteractive apt install -y python3-opencv
RUN pip install timm==0.4.12
RUN cd src/model/deformconv/ && python setup.py build install
# RUN apt install -y wget unzip
# ARG COMMIT=4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
# RUN wget https://github.com/NVIDIA/apex/archive/${COMMIT}.zip && unzip ${COMMIT}.zip && \
#     rm ${COMMIT}.zip && cd apex-${COMMIT} && \
#     pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list

RUN apt update --allow-unauthenticated --allow-insecure-repositories && \
    DEBIAN_FRONTEND=noninteractive apt install -y python3-opencv
RUN pip install imageio


#     DOCKER_BUILDKIT=0 docker build -t cf --network=host .
