FROM python:3.11.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# SCTools Docker SBL 26/11/2024
WORKDIR /app
COPY main.py .
COPY requirements.txt .
RUN set -eux; \
    pip uninstall -y opencv-python; \
    pip install --no-cache-dir opencv-python-headless -i https://mirrors.aliyun.com/pypi/simple \
    pip install -r requirements.txt 

    EXPOSE 9003

COPY best_openvino_model /app/best_openvino_model

CMD ["python", "./main.py -ip 0.0.0.0 -p 9003 -workers 2"]
