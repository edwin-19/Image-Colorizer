# Image Coloring 
A project to color a black and white image to color using deep learning

# How to run
First install the neccessary libraries
```python
pip install -r requirements.txt
```

Download the model and run script
```python
python demo_autoencoder.py -p -m model/effnet_colorizer.h5 -i data/path/to/image
```
or pix2pix (generally more accurate)

```python
python demo_pix2pix.py -i data/path/to/image
```

# Intution 
The idea on how to convert an image from black white to a color image can be broken down to the following steps:
1) Read Image as RGB
2) Convert Image to Lab Color Space
3) Create neural network to map L color space to ab color space
4) Model will use L (black white) to predict ab color space
5) Create empty image and add L image and ab image to form a complete Lab Image
6) Convert Lab Image to RGB again to get back a colorize image

# Results
![image_results](https://github.com/edwin-19/Image-Colorizer/blob/master/results/results.jpg?raw=true)

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
- [x] Upload model drive download links
- [x] Requirements txt
- [x] Add Demo Results
- [ ] Write out findings
- [ ] Try to move to specified architecture using nvidia triton
- [ ] Move preporcessing with a model instead of code 

# Notes
- As of the current available models pretrained model with autoencoder seems to have better results
- Previously though of metrics where eyeballing is unavoidable has changed a bit
- Need to switch to pytorch for gan model as seems easier to read for now


# Models
You can download the models trained here
| Model Name                | Backbone         | Drive Link                                                                                 |
|---------------------------|------------------|--------------------------------------------------------------------------------------------|
| Simple AutoEncoder        | None             | [here](https://drive.google.com/file/d/1E9eRsd1rS2hMTbU9viQD46bTthgGvC01/view?usp=sharing) |
| EffNet Autoencoder        | Efficient Net B3 | [here](https://drive.google.com/file/d/1ChfDyZmpxAGnZTR-WVbYrnzqiPPRBZL0/view?usp=sharing) |
| Unet Pretrained Generator | Resnet 18        | [here](https://drive.google.com/file/d/12IKcMlcCghat8qTbemQLNtEDGKLrlatF/view?usp=sharing) |
| Unet Finetuned Generator  | Resnet 18        | [here](https://drive.google.com/file/d/1VfZJb5iKdxG4_udOslEJvpWAZUnagQts/view?usp=sharing) |

# Evaluation Score
| Model              | DISTS Score |
|--------------------|-------------|
| Pix2pix - Resnet18 | 0.7836      |
|                    |             |

# References
The following repos were based off these guides
- https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d
- https://nbviewer.jupyter.org/github/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/Image%20Colorization%20with%20U-Net%20and%20GAN%20Tutorial.ipynb
- https://github.com/emilwallner/Coloring-greyscale-images
- https://towardsdatascience.com/u-net-deep-learning-colourisation-of-greyscale-images-ee6c1c61aabe
