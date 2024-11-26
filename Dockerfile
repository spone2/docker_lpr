FROM python:3.11.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# SCTools Docker SBL 26/11/2024
WORKDIR /app

RUN set -eux; \
    pip uninstall -y opencv-python; \
    pip install --no-cache-dir opencv-python-headless -i https://mirrors.aliyun.com/pypi/simple

    EXPOSE 9003

COPY best_openvino_model /app/best_openvino_model

CMD ["bash", "-c", "lpr_eu_api -ip 0.0.0.0 -p 9003 -workers 2"]
