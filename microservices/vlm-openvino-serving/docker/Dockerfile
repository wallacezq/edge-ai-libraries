# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12.12-slim AS build

RUN pip install --upgrade --no-cache-dir pip setuptools

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.create false && poetry install --no-root --only=main

FROM python:3.12.12-slim

ARG INSTALL_DRIVER_VERSION="26.09.37435"
ARG USER_ID=1000
ARG USER_GROUP_ID=1000

RUN groupadd -g ${USER_GROUP_ID} appuser && useradd -m -s /bin/bash -u ${USER_ID} -g ${USER_GROUP_ID} appuser

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    libpng16-16t64 libpng-dev ffmpeg libjpeg-dev \
    libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev git \ 
    && rm -rf /var/lib/apt/lists/* 

#RUN ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so.62 /usr/lib/x86_64-linux-gnu/libjpeg.so.8
RUN curl http://archive.debian.org/debian/pool/main/libj/libjpeg8/libjpeg8_8b-1_amd64.deb -o /tmp/libjpeg8_8b-1_amd64.deb && \
    apt-get install -y --no-install-recommends --fix-missing /tmp/libjpeg8_8b-1_amd64.deb && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app/poetry.lock /app/poetry.lock

COPY ./scripts/install_ubuntu_gpu_drivers.sh /tmp/install_gpu_drivers.sh
RUN chmod +x /tmp/install_gpu_drivers.sh
RUN /tmp/install_gpu_drivers.sh

RUN apt-get clean && rm -rf /tmp/install_gpu_drivers.sh && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/* && rm -rf /root/.cachetmp/*

# Optionally download copyleft sources if requested
ARG COPYLEFT_SOURCES=true
RUN if [ "$COPYLEFT_SOURCES" = "true" ]; then \
        apt-get update && \
        # Get list of installed deb packages with copyleft licenses \
        sed -Ei 's/# deb-src /deb-src /' /etc/apt/sources.list && \
        apt-get update && \
        mkdir -p /copyleft_sources/deb && cd /copyleft_sources/deb && \
        echo -n $null > copyleft_package_list.txt && \
        for package in $(dpkg -l | awk '/^ii/ {print $2}' | cut -d: -f1); do \
            grep -l 'Copyleft\|GPL\|LGPL\|EPL\|MPL\|CDDL' /usr/share/doc/${package}/copyright; \
            exit_status=$?; \
            if [ $exit_status -eq 0 ]; then \
                echo $package >> copyleft_package_list.txt; \
                apt-get source -q --download-only $package; \
            fi; \
        done; \
        # Get source code for installed Python packages with copyleft licenses \
        mkdir -p /copyleft_sources/python && \
        cd /copyleft_sources/python && \
        apt-get update && apt-get install --no-install-recommends -y gcc build-essential libffi-dev python3-dev && \
        # Download python package sources with relevant licenses \
        pip3 freeze | cut -d= -f1 | while read pkg; do \
            meta=$(pip3 show $pkg 2>/dev/null); \
            lic=$(echo "$meta" | grep -i '^License:' | grep -E 'MPL|GPL|General Public License|EPL|Eclipse Public License|CDDL|LGPL'); \
            if [ ! -z "$lic" ]; then \
                echo "Downloading source for $pkg with license: $lic"; \
                pip3 download --no-binary :all: $pkg || true; \
            fi; \
        done; \
        # apt-get remove --purge -y gcc build-essential libffi-dev python3-dev; \
    fi
    
WORKDIR /app
COPY . .

#RUN pip install -U /app/torchvision-0.23.0a0+824e8c8-cp312-cp312-linux_x86_64.whl
#RUN git clone --branch v0.23.0 https://github.com/pytorch/vision.git && \
#   cd vision 

RUN export BUILD_VIDEO_READER=1\
    export BUILD_TORCHVISION_OPS=0\
    export FORCE_CUDA=0\
    export NO_CUDA=1

RUN pip install "setuptools<82"

RUN git clone --branch v0.23.0 https://github.com/pytorch/vision.git && \
    cd vision && \
    python setup.py build_ext --inplace
RUN cp /app/vision/torchvision/video_reader.so /usr/local/lib && \
    cp /app/vision/torchvision/video_reader.so /usr/local/lib/python3.12/site-packages/torchvision/

RUN rm -rf vision
RUN apt-get remove --purge -y gcc build-essential libffi-dev python3-dev

RUN chmod +x /app/scripts/compress_model.sh

RUN mkdir -p /home/appuser/.cache/huggingface /app/ov-model && \
    chown -R appuser:appuser /home/appuser/.cache /app/ov-model /opt

USER appuser

EXPOSE 8000

CMD ["/bin/bash", "-c", "/app/scripts/compress_model.sh ${VLM_MODEL_NAME} ${VLM_COMPRESSION_WEIGHT_FORMAT} ${HUGGINGFACE_TOKEN} && gunicorn -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker src.app:app --timeout 10000 -w ${WORKERS:-1} --log-level ${VLM_OPENVINO_LOG_LEVEL:-info} --access-logfile ${VLM_ACCESS_LOG_FILE:--} --error-logfile -"]
