from fastapi import FastAPI, File, UploadFile
import numpy as np
from turbojpeg import TurboJPEG
from skimage.color import rgb2lab, lab2rgb

app = FastAPI()
turbojpeg = TurboJPEG()

import onnxruntime

sess_options = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession('model/color_gen.onnx', sess_options)

@app.post('/')
async def get_index(file: UploadFile = File(...)):
    contents = await file.read()
    img_buffer = np.frombuffer(contents, dtype=np.uint8)
    img = turbojpeg.decode(img_buffer)

    lab_img = rgb2lab(img).astype(np.float32)
    L = lab_img[:,:, 0] / 50. - 1.
    L = np.expand_dims(L, axis=0)
    L = np.expand_dims(L, axis=-1)
    
    pred_ab = sess.run([output_name], {
        input_name: np.transpose(L, (0, 3, 1, 2))
    })[0]
    
    
    # Post processing
    pred_ab = np.squeeze(pred_ab, axis=0) * 110

    pred_ab = np.transpose(pred_ab, (1, 2, 0))

    empty_img = np.zeros((256, 256, 3))
    empty_img[:, :, 0] = lab_img[:, :, 0]
    empty_img[:, :, 1:] = pred_ab
    rgb_pred = lab2rgb(empty_img)
    
    return {
        'img': turbojpeg.encode(rgb_pred).decode('ISO-8859-1')
    }