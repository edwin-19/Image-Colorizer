# Image Coloring 
A project to color a black and white image to color using deep learning

# Intution 
The idea on how to convert an image from black white to a color image can be broken down to the following steps:
1) Read Image as RGB
2) Convert Image to Lab Color Space
3) Create neural network to map L color space to ab color space
4) Model will use L (black white) to predict ab color space
5) Create empty image and add L image and ab image to form a complete Lab Image
6) Convert Lab Image to RGB again to get back a colorize image

# ToDO
- [ ] Train AutoEncoder with Fusion of Pretrained Model
- [ ] Train pix2pix(GAN) model for more accurate results
- [ ] Add metrics specified [here](https://arxiv.org/pdf/2008.10774.pdf)
- [ ] If time is available move to deployment architecture (FAST API)