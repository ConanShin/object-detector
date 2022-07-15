from fastapi import FastAPI, File
from interface.image_url_interface import ImageUrlInterface
from model import CarCounterModel

app = FastAPI()
model = CarCounterModel()


@app.get("/")
async def root():
    return {
        'status': 'ok'
    }


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
async def detect_from_image_file(file: bytes = File()):
    loaded_image = await model.load_image_file(file)
    features = model.feature_extraction(loaded_image)
    probas, keep, bboxes = model.softmax(loaded_image, features)
    count = model.count_object(probas[keep], 'car')
    return {
        'total object count': bboxes.shape[0],
        'car count': count
    }