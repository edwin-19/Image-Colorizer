from fastapi import FastAPI, File, UploadFile
import numpy as np
from turbojpeg import TurboJPEG

app = FastAPI()
turbojpeg = TurboJPEG()

@app.post('/')
async def get_index(file: UploadFile = File(...)):
    contents = await file.read()
    img_buffer = np.frombuffer(contents, dtype=np.uint8)
    img = turbojpeg.decode(img_buffer)
    
    return {
        'img': str(img_buffer)
    }