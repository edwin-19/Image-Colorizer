from fastapi import FastAPI, File, UploadFile
import numpy as np
from turbojpeg import TurboJPEG
from skimage.color import rgb2lab, lab2rgb

app = FastAPI()
turbojpeg = TurboJPEG()

@app.post('/')
async def get_index(file: UploadFile = File(...)):
    contents = await file.read()
    img_buffer = np.frombuffer(contents, dtype=np.uint8)
    img = turbojpeg.decode(img_buffer)

    lab_img = rgb2lab(img).astype(np.float32)
    L = lab_img[[0], ...] / 50. - 1. # Between -1 and 1
    
    return {
        'img': str(img_buffer)
    }