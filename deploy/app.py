import io
import csv
import cv2
from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    HTTPException, 
    status,
    Depends
)
from fastapi.responses import Response, StreamingResponse
import numpy as np
from PIL import Image
from predictor import CursoPredictor, FaceDetector
from datetime import datetime as dt
import time

processed_image = None
processed_image_info = None

app = FastAPI(title="Curso reconocedor")

predictor = CursoPredictor()

face_detector = FaceDetector()

def get_predictor():
    return predictor

def get_face_detector():
    return face_detector

def predict_uploadfile(predictor, file):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="No es una imagen"
        )
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array), img_array

@app.get("/status")
def get_status():
    model_info = {
        "model_name": "Curso Predictor",
        "version": "1.0",
        "status": "en linea",
        "author": "Pablo Badani"
    }

    service_info = {
        "service_name": "Curso reconocedor API",
        "status": "online"
    }

    return {
        "model_info": model_info,
        "service_info": service_info
    }

@app.post("/annotate", responses={
    200: {"content": {"image/jpeg": {}}}
})
def predict_and_annotate(
    file: UploadFile = File(...), 
    predictor: CursoPredictor = Depends(get_predictor)
) -> Response:
    global processed_image
    global processed_image_info
    results, img = predict_uploadfile(predictor, file)
    processed_image_info = {
        "file_name": file.filename,
        "results": results,
        "current_datetime": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time": None,
        "model": "Curso Predictor"
    }
    processed_image = img
    new_img = cv2.putText(
        img,
        f"{results['class']} - Confidence: {results['confidence']:.2f}%",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    img_pil = Image.fromarray(new_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/faces", responses={
    200: {"content": {"image/jpeg": {}}}
})
def detect_faces(
    file: UploadFile = File(...), 
    predictor: FaceDetector = Depends(get_face_detector)
) -> Response:
    results, img = predict_uploadfile(predictor, file)
    for result in results:
        bbox = result['bbox']
        keypoints = result['keypoints']
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                            (0, 255, 0), 2)
        for i in range(2):  
            x = int(keypoints[i][0] * img.shape[1])  
            y = int(keypoints[i][1] * img.shape[0])  
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
        if len(keypoints) >= 3:  
            x = int(keypoints[2][0] * img.shape[1]) 
            y = int(keypoints[2][1] * img.shape[0]) 
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
        if len(keypoints) >= 4:  
            x = int(keypoints[3][0] * img.shape[1])  
            y = int(keypoints[3][1] * img.shape[0])  
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
    img_pil = Image.fromarray(img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.get("/reports")
def generate_report(
    predictor: CursoPredictor = Depends(get_predictor),
):
    global processed_image
    if processed_image is None:
        return {"error": "No se ha procesado ninguna imagen aún"}
    start_time = time.time()
    results = predictor.predict_image(processed_image)
    processed_image_info["execution_time"] = time.time() - start_time
    current_datetime = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    execution_time = time.time() - start_time
    with open("report.csv", mode="a", newline="") as csvfile:
        fieldnames = [
            "Nombre del archivo",
            "Prediccion",
            "Fecha",
            "Execution Time",
            "Modelo",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(
        {
            "Nombre del archivo": processed_image_info["file_name"],
            "Prediccion": f"{processed_image_info['results']['class']} - Confidence: {processed_image_info['results']['confidence']:.2f}%",
            "Fecha": processed_image_info["current_datetime"],
            "Execution Time": f"{processed_image_info['execution_time']:.4f} seconds",
            "Modelo": processed_image_info["model"],
        }
    )
    with open("report.csv", mode="r", newline="") as csvfile:
        content = csvfile.read()
        return StreamingResponse(io.StringIO(content), media_type="text/csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)