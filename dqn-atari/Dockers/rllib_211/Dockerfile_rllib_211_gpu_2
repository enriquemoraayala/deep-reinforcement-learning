FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker-v1.14

# Dependencias de sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends swig build-essential && \
    rm -rf /var/lib/apt/lists/*

# Asegura que el swig usado sea el binario del sistema (no el wrapper de pip)
RUN rm -f /opt/conda/bin/swig && ln -s /usr/bin/swig /opt/conda/bin/swig && swig -version

# Pip toolchain
RUN pip install --upgrade pip setuptools wheel cython numpy

# Resto deps
RUN pip install pyarrow==15.0.2 \
    ray==2.11.0 "ray[rllib]==2.11.0"

# Box2D (siempre compila desde source en py311 muchas veces)
RUN pip install box2d-py==2.3.5

RUN pip install matplotlib==3.8.4 imageio==2.34.0 pillow==10.3.0 debugpy

RUN pip install gymnasium==1.0.0
RUN pip install "gymnasium[box2d]"

WORKDIR /opt/ml/code

ENV PYTHONUNBUFFERED=TRUE

