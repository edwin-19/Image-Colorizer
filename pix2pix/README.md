# Pix2pix model for color conversion
- Same principle as the autoencoder version but with the added discriminatior to help differentiate

Note:
- The reason for such poor accuracy was due to the loss function, when its not very confident it will default most colors to a brownish color
- For the unet generator there are 2 types:
    - Handwritten from sratch
    - Fastai with pretrained network

# Training code
Train pix2pix.ipynb
- Contains 2 parts to train you most likely train one part first restart and train the other network
    - First part is the generator (unet) with a pretrained network (resnet18)
    - Second part requires you to load the pretrained generator from earlier and trained along as a gan

# Conversion script
- Here contains a script to convert the resnet model to onnx for deployment inference