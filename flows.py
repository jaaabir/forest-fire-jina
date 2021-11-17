from jina import Document, DocumentArray, Flow, Executor, requests
from typing import Dict
import tensorflow as tf
import numpy as np 
import cv2


def Loadmodel():
    model_exp = tf.keras.models.load_model('exp_cnn.h5')
    print('''
    
    ###################################################################################
    
                USING THE EXPERIMENTAL MODEL DUE TO SECURITY REASONS

    ###################################################################################
    
    
    ''')
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


with flow as f:
    f.block()
