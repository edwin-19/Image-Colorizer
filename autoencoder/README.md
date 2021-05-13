# AutoEncoder
This side of the repo contains all the training code for the autoencoder for image colorization

Note:
- The model was written using tensorflow 2.3
- For the color corization tensorflowio(0.16.0) was used so that it can work with tensorflow datasets
- The article referenced used here used inceptionresnetv2, this repo uses efficientnet b3(https://github.com/qubvel/efficientnet)

# Training code
- SimpleAutoEncoder.ipynb
    - Follows a simple autoencoder style
    - Structure is as below for model architeture
    - ![alt](https://miro.medium.com/max/700/1*xEj4c-CGzXe2Zh3BM1IdZQ.png?raw=True)
    - Encoder layers:
        - Summarizes and extracts information from the image
    - Decoder layers:
        - Colors and resize the picture back the image to its original image

- Transfer Learning.ipnb
    - Follows the architecture above but adds a pretrained model for classification
    - Concats the input from the encoder and classifier
    - Instead of inception resnetv2, i have swithced to efficientnet b3
    - ![alt](https://miro.medium.com/max/700/1*KRXxAAxlBz1psRvB1ak04Q.png?raw=True)

Note: You might need to configure the training directory, changed when moving files within the repo