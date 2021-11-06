from typing import Dict
from jina import Document, DocumentArray, Flow, Executor, requests
import matplotlib.pyplot as plt 
import cv2
import numpy as np 
import tensorflow as tf
import os


def Loadmodel():
    # model_exp = tf.keras.models.load_model('model_dnet121.h5')
    print('''
    
    ########################################################
    
                USE YOUR OWN DL MODEL
                
    ########################################################
    
    
    ''')
    model_exp = None
    return model_exp


def resizer(docs):

    X = DocumentArray()
    for doc in docs:
        resize_blob = cv2.resize(doc.blob, dsize = (256, 256), interpolation= cv2.INTER_CUBIC)
        resize_blob = resize_blob.reshape(256, 256, 3)
        resize_blob = resize_blob / 255
        n_doc = Document(blob = resize_blob, tags = doc.tags)
        X.append(n_doc)

    return X

class Classify(Executor):
    @requests
    def predict(self, docs : DocumentArray, parameters : Dict, **kwargs):
        X = np.array([
            i.blob for i in resizer(docs)
        ])
        model_exp = Loadmodel()
        prediction = (model_exp.predict(X) > 0.41).astype(int)
        prediction = prediction.reshape(1, -1)[0]

        pred_docs = DocumentArray()
        for ind, doc in enumerate(docs):
            # print(prediction[ind])
            doc.tags['class'] = str(prediction[ind])
            print(doc.tags['class'])
            pred_docs.append(doc)

        return pred_docs 
        

flow = (
    Flow(cors = True, protocol = 'http', install_requirements=True, port_expose = 12345)
    .add(
        name = 'fire_detector',
        uses = Classify
    )
)

doc = Document(uri = 'images/apple.png')
doc.convert_uri_to_image_blob()
testing = DocumentArray([doc])

with flow as f:
    f.post(
        on = '/index',
        inputs = testing,
        on_done = lambda x : x.docs[0].tags
    )

    f.block()





# class ImageResizer(Executor):
#     @requests
#     def resize(self, docs : DocumentArray, parameters : Dict, **kargs):
#         resize_docs = DocumentArray()
#         for doc in docs:
#             resize_blob = cv2.resize(doc.blob, dsize = (256, 256), interpolation= cv2.INTER_CUBIC)
#             resize_blob = resize_blob.reshape(-1, 256, 256, 3)
#             resize_blob = resize_blob / 255
#             new_doc = Document(blob = resize_blob, tags = doc.tags)
#             resize_blob.append(new_doc)

#         return resize_docs



# class ImageNormalizer(Executor):
#     @requests
#     def normalize(self, docs : DocumentArray, parameters : Dict, **kargs):
#         norm_docs = DocumentArray()
#         for doc in docs:
            
#             img_blob = doc.blob / 255
#             new_doc = Document(blob = img_blob, tags = doc.tags)
#             norm_docs.append(new_doc)

#         return norm_docs