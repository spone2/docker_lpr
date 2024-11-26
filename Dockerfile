FROM python:3.11.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# SCTools Docker SBL 26/11/2024
WORKDIR /app
COPY main.py ./app/main.py
COPY requirements.txt ./app/requirements.txt
COPY best_openvino_model /app/best_openvino_model

RUN set -eux;
RUN pip uninstall -y opencv-python;
RUN pip install --no-cache-dir opencv-python-headless -i https://mirrors.aliyun.com/pypi/simple;
RUN pip install -r ./app/requirements.txt;

EXPOSE 9003



CMD ["python", "main.py -ip 0.0.0.0 -p 9003 -workers 2"]
