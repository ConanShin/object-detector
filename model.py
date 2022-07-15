import io

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests
import torch


class CarCounterModel:
    def __init__(self):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        print('model loaded')

    @staticmethod
    def load_image(url):
        image = Image.open(requests.get(url=url, stream=True).raw)
        print('image size: ', image.size)
        return image

    @staticmethod
    async def load_image_file(file: bytes):
        image = Image.open(io.BytesIO(file))
        print('image size: ', image.size)
        return image

    def feature_extraction(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return self.model(**inputs)

    def softmax(self, image, outputs):
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)

        postprocessed_outputs = self.feature_extractor.post_process(outputs, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
        print('total detected object count: ', bboxes_scaled.shape[0])
        return probas, keep, bboxes_scaled

    def count_object(self, probas_keep, category='car'):
        count = 0
        for p in probas_keep:
            cl = p.argmax()
            if self.model.config.id2label[cl.item()] == category:
                count = count + 1
        print('total ', category, ' count: ', count)
        return count
