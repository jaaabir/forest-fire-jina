from typing import Dict
from jina import Document, DocumentArray, Flow, Executor, requests
import matplotlib.pyplot as plt 
import cv2
import os


class ImageCleaner(Executor):
    @requests
    def clean(self, docs : DocumentArray, parameters : Dict, **kargs):
        blurred_docs = DocumentArray()
        if docs:
            for doc in docs:
                blur = cv2.blur(doc.blob, (5,5))
                new_doc = Document(blob = blur, tags = doc.tags)
                blurred_docs.append(new_doc)

        return blurred_docs


class Classify(Executor):
    @requests
    def predict(self, docs : DocumentArray, parameters : Dict, **kwargs):
        pass

flow = (
    Flow(cors = True, protocol = 'http', install_requirements=True, port_expose = 12344)
    .add(
        name = 'image_normalizer',
        uses ='jinahub://ImageNormalizer',
    )
    .add(
        name = 'image_noice_remover',
        uses = ImageCleaner
    )
    .add(
        name = 'image_segmenter',
        uses = 'jinahub://YoloV5Segmenter',
    )
    # .add(
    #     name = 'fire_detector',
    #     uses = Classify
    # )
)

testing = DocumentArray([])

with flow as f:
    f.post(
        on = '/index',
        inputs = testing,
        on_done = print
    )

    f.block()