import tensorflow as tf
import tensorflow_io as tfio
from efficientnet.tfkeras import EfficientNetB3

import numpy as np
from matplotlib import pyplot as plt
import utils
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i','--image', type=str, default='data/images/Test/xkObNU.jpg'
    )
    
    parser.add_argument(
        '-m', '--model', default='model/color_autoencode.h5'
    )
    parser.add_argument(
        '-p', '--pretrained_model', default=False, action="store_true"
    )
    
    args = parser.parse_args()
    
    image_byte = tf.io.read_file(args.image)
    img = tf.io.decode_jpeg(image_byte)
    
    model = tf.keras.models.load_model(args.model, compile=False)
    
    img_norm = tf.cast(img, tf.float32)
    img_norm = img_norm / 255.
    lab_img = tfio.experimental.color.rgb_to_lab(img_norm)
    l_img = lab_img[:, :, 0]
    
    if args.pretrained_model:
        effnet_model = EfficientNetB3(weights='imagenet', include_top=True)
        img_resized = tf.image.resize(img, (300, 300))
        img_resized = img_resized / 255.
        embeddings = effnet_model.predict(tf.expand_dims(img_resized, axis=0))
        
        ab_pred = model.predict([tf.expand_dims(l_img, axis=0), embeddings])
    else:
        ab_pred = model.predict(tf.expand_dims(l_img, axis=0))
    
    lab_img_recon = np.zeros((256, 256, 3))
    lab_img_recon[:, :, 0] = l_img
    lab_img_recon[:, :, 1:] = np.squeeze(ab_pred * 128, axis=0)
    
    utils.visualize(
        ori_img=img, L=l_img,
        pred_color=tf.cast(tfio.experimental.color.lab_to_rgb(lab_img_recon)* 255, tf.uint8) 
    )