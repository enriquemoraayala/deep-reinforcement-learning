# Dockerfile
# FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04
FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker-v1.14

# Instalar dependencias de sistema
RUN apt-get update && \
    apt-get install -y swig build-essential python3-dev

# Actualiza pip y wheel
RUN pip install --upgrade pip setuptools wheel cython numpy \
    && pip install swig \
    && pip install pyarrow==15.0.2 \
    && pip install ray==2.11.0 \
    && pip install ray[rllib]==2.11.0 

RUN pip install box2d-py==2.3.5 
RUN pip install matplotlib==3.8.4 \
    && pip install imageio==2.34.0 \
    && pip install pillow==10.3.0 \
    && pip install debugpy

RUN pip install gymnasium==1.0.0
RUN pip install gymnasium[box2d] 

# RUN mkdir -p /opt/ml/code /opt/ml/output
# (Opcional) Añade tus scripts
# COPY train_lunar_lander_rllib_aws.py /opt/ml/code/train_lunar_lander_rllib_aws.py
# COPY gen_eps_lunar_lander_rllib_ppd.py /opt/ml/code/gen_eps_lunar_lander_rllib_ppd.py
# COPY process_frames.py /opt/ml/code/process_frames.py

WORKDIR /opt/ml/code

ENV PYTHONUNBUFFERED=TRUE