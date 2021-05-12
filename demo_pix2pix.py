from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms as T
import torch
import cv2
import os
import utils
from matplotlib import pyplot as plt
import numpy as np

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i','--image', type=str, default='data/images/Test/xkObNU.jpg'
    )
    
    parser.add_argument(
        '-m', '--model', default='model/color_gen.pt'
    )
    args = parser.parse_args()
    
    # Load model
    body = create_body(resnet18, pretrained=True, n_in=1, cut=-2)
    net_G = DynamicUnet(body, 2, (256, 256)).to('cuda')
    net_G.load_state_dict(torch.load(args.model))
    net_G.to('cuda')
    net_G.eval()
    
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lab_img = rgb2lab(img).astype(np.float32)
    lab_img = T.ToTensor()(lab_img)
    L = lab_img[[0], ...] / 50. - 1. # Between -1 and 1
    
    with torch.no_grad():
        pred_ab = net_G(L.unsqueeze(0).to('cuda'))
    
    L_true = lab_img[[0], ...]
    pred_ab = pred_ab * 110.
    Lab_pred = torch.cat([L_true, pred_ab.squeeze(0).cpu()])

    Lab_pred = Lab_pred.permute(1, 2, 0).numpy()
    utils.visualize(
        orginal=img,
        L=L_true.permute(1, 2, 0).numpy(), color=lab2rgb(Lab_pred)
    )