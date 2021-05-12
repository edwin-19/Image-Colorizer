# Image Coloring 
A project to color a black and white image to color using deep learning

# How to run


# Intution 
The idea on how to convert an image from black white to a color image can be broken down to the following steps:
1) Read Image as RGB
2) Convert Image to Lab Color Space
3) Create neural network to map L color space to ab color space
4) Model will use L (black white) to predict ab color space
5) Create empty image and add L image and ab image to form a complete Lab Image
6) Convert Lab Image to RGB again to get back a colorize image

# ToDO
- [x] Train AutoEncoder with Fusion of Pretrained Model
- [x] Add metrics specified [here](https://arxiv.org/pdf/2008.10774.pdf)
- [x] Inference script for all models
- [x] Train pix2pix(GAN) model for more accurate results
- [x] Train with dynamic unet for pix2pix
- [x] Use pytorch image quality metrics for image quality assements
- [x] Converted model to onnx for deployment
- [x] Onnx Inference Script
- [x] If time is available move to deployment architecture (FAST API)

# ToDO Clean Up
- [x] Upload model
- [x] Requirements txt
- [ ] Write out findings
- [ ] Try to move to specified architecture using nvidia triton
- [ ] Move preporcessing with a model instead of code 

# Notes
- As of the current available models pretrained model with autoencoder seems to have better results
- Previously though of metrics where eyeballing is unavoidable has changed a bit
- Need to switch to pytorch for gan model as seems easier to read for now


# Models
You can download the models trained here


# Evaluation Score
| Model              | DISTS Score |
|--------------------|-------------|
| Pix2pix - Resnet18 | 0.7836      |
|                    |             |