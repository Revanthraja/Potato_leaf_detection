from fastapi import FastAPI,File,UploadFile
import uvicorn 
import numpy as np
from  io import *
from PIL import Image
import tensorflow as tf
app=FastAPI()
Model=tf.keras.models.load_model('/media/revanth/Data/Jupyter/potato_leaf_detection/potato.h5')
class_names=['Early_Blight', 'Healthy', 'Late_Blight']

@app.get("/ping")
async def ping():
    return "hello world"

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
    file:UploadFile = File(...)
):
    image=read_file_as_image(await file.read())
    imag_batch=np.expand_dims(image,0)
    prediction=Model.predict(imag_batch)
    predicted_class=class_names[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])

    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }

    





if __name__ =="__main__":
    uvicorn.run(app,host='localhost',port=8000)