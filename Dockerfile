FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
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


RUN pip install timm==0.4.12
RUN cd src/model/deformconv/ && python setup.py build install


RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list

RUN apt update --allow-unauthenticated --allow-insecure-repositories && \
    DEBIAN_FRONTEND=noninteractive apt install -y python3-opencv && \
    apt-get install -y tmux

RUN pip install imageio
RUN pip install open3d


#     DOCKER_BUILDKIT=0 docker build -t cf --network=host .


