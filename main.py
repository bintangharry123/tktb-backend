import time
from fastapi import FastAPI, status, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import onnxruntime as ort

class ImageInput(BaseModel):
  image: str

app = FastAPI()

CLASSES = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion' ]

file_type = [
    'image/png',
    'image/jpeg',
    'image/jpg'
]

@app.get('/')
def get_root():
  return {"massage": "Hello"}

@app.get('/tktb1')
def get_root():
  return {"message": "halo dari tktb1"}

def prepare(byte_image):
  img = Image.open(io.BytesIO(byte_image)).resize((256,256))
  img_arr = np.array(img)
  if img_arr.ndim == 2:
    img_arr = np.stack([img_arr] * 3 , axis=-1)

  img_arr = img_arr.astype(np.float32) / 255.0
  img_arr = np.transpose(img_arr, (2, 0, 1))
  img_arr = np.expand_dims(img_arr, axis=0)

  return img_arr


@app.post('/get-prediction')
def predict(file: UploadFile):
    try:
        file_type.index(file.content_type)
    except ValueError :
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Cannot process wrong file format. Please send these type of file format: [jpeg/jpg/png]',
                            headers={
                                "Accept": "image/jpeg, image/jpg, image/png"
                            })
    img = file.file.read()

    tensor = prepare(img)
    
    ort_session = ort.InferenceSession("model.onnx")

    try:
      ort_inputs = {ort_session.get_inputs()[0].name: tensor}
      ort_outs = ort_session.run(None, ort_inputs)
      index = np.array(ort_outs).argmax()
      return {"result": CLASSES[index]}
    except Exception as e:
      raise HTTPException(400, "Bad Request")

