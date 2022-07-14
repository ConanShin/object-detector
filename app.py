import uvicorn
from fastapi import FastAPI, File, UploadFile
from interface.image_url_interface import ImageUrlInterface
from model import CarCounterModel

app = FastAPI()
model = CarCounterModel()


@app.get("/health")
async def health_check():
    return {
        'status': 'ok'
    }


@app.post("/url")
async def detect_from_image_link(request_body: ImageUrlInterface):
    if len(request_body.url) == 0:
        return 'invalid url'
    loaded_image = model.load_image(request_body.url)
    features = model.feature_extraction(loaded_image)
    probas, keep, bboxes = model.softmax(loaded_image, features)
    count = model.count_object(probas[keep], 'car')
    return {
        'total object count': bboxes.shape[0],
        'car count': count
    }


@app.post("/file")
async def detect_from_image_file(file: UploadFile = File()):
    loaded_image = await model.load_image_file(file)
    features = model.feature_extraction(loaded_image)
    probas, keep, bboxes = model.softmax(loaded_image, features)
    count = model.count_object(probas[keep], 'car')
    return {
        'total object count': bboxes.shape[0],
        'car count': count
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)