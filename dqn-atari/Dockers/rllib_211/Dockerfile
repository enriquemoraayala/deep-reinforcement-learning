# Dockerfile (GPU + CUDA)
FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker-v1.14

RUN pip install --upgrade pip \
    && pip install swig \
    && pip install pyarrow==15.0.2 \
    && pip install box2d-py==2.3.5 \
    && pip install ray==2.11.0 \
    && pip install ray[rllib]==2.11.0 \
    && pip install gymnasium==0.28.1 \
    && pip install gymnasium[box2d] \
    && pip install matplotlib==3.8.4 \
    && pip install imageio==2.34.0 \
    && pip install pillow==10.3.0 \
    && pip install debugpy

WORKDIR /opt/ml/code
ENV PYTHONUNBUFFERED=TRUE
