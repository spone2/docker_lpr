import sys
import cv2
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import logging
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile
import argparse
import uvicorn

sys.path.append(str(Path(__file__).resolve().parent.parent))
#valores por defecto de ajuste
conf = 0.5

#versión del módulo
sctools_version ="0.0.1_261124"
logging.info('sctools %s iniciado.',sctools_version)

#Lo primero es cargar el modelo entrenado para placas europeas
ov_model = YOLO("best_openvino_model/")
logging.info('Modelo cargado')

#Nombre de la aplicación FastAPI
app = FastAPI(title='SCTools-Core')
#EndPoint Base
@app.get("/")
def root():
    return {"message": "Bienvenido a SCTools Server"}

@app.post("/ocr")
def ocr(
    image_file: UploadFile = None,
    multi_plate: bool = Form(None),
):
    if image_file:
        #img = Image.open(image_file.file)
        img = cv2.imread(image_file.file,1)
    elif image_data:
        img_bytes = str.encode(image_data)
        img_b64decode = base64.b64decode(img_bytes)
        img = Image.open(io.BytesIO(img_b64decode))
    else:
        raise ValueError(
            "No se ha enviado imagen o datos."
        )
    
    #ya tenemos la imagen, vamos a hacer la inferencia para obtener la matrícula.
    placas_img = read_plates(img)
    resultado_ocr = read_plate_number_vino(placas_img) 
    return resultado_ocr

#obtenemos las placas de matrícula
def read_plates(img):
    #Esta primera versión obtiene la primera placa nada más
    logging.info('Buscando matrículas en la imagen.')
    results = ov_model(img,conf=0.5)
    
    #cogemos la inferencia más alta (revisar código)
    xyxy = results[0].boxes.xyxy.numpy()
    
    #coordenadas de la matrícula
    x1, y1 = int(xyxy[0][0]), int(xyxy[0][1])
    x2, y2 = int(xyxy[0][2]), int(xyxy[0][3])

     # Expand Region for OCR to get more background (helps with detection)
    offset = 0.2
    x1 = int(x1*(1-offset))
    y1 = int(y1*(1-offset))
    x2 = int(x2*(1+offset))
    y2 = int(y2*(1+offset))

    img_rec = img[y1:y2, x1:x2]
    #devolvemos la imagen de la placa
    return img_rec
    
#Hacemos OCR de la imagen de la placa
def read_plate_number_vino(img):
    engine = RapidOCR()
    result, elapse = engine(img)
    boxes, txts, scores = list(zip(*result))
    data = result
    # Convert to a dictionary with appropriate keys
    output = {
        'result': data,
        'elapse': elapse,
        'words': txts,
        'boxes':boxes,
        'scores':scores
    }
    return output
    # Write to a JSON file
    #with open('output.json', 'w') as json_file:
    #    json.dump(output, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser("sctools_api")
    parser.add_argument("-ip", "--ip", type=str, default="0.0.0.0", help="IP Address")
    parser.add_argument("-p", "--port", type=int, default=9003, help="IP port")
    parser.add_argument(
        "-workers", "--workers", type=int, default=1, help="number of worker process"
    )
    args = parser.parse_args()

    uvicorn.run("sctools_api.main:app",host=args.ip,port=args.port,reload=0,workers=args.workers)

if __name__ == '__main__': 
    # train()
    main()