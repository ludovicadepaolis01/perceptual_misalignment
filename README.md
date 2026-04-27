This is the code for "Perceptual misalignment of texture representations in convolutional neural networks" by de Paolis et al.

- **Requirements**
  ```text
  torch = 2.2.0a0+git6c8c5ad
  torchvision = 0.17.0+b2383d4
  numpy = 1.26.4
  scipy = 1.12.0
  pandas = 2.2.1
  matplotlib = 3.8.3
  Pillow (PIL) = 10.2.0
  h5py = 3.10.0
  regex = 2023.12.25
  natsort = 8.4.0
  ndd = 1.10.6
  scikit-learn = 1.4.1.post1
- **Data**  
  _Describable Textures Dataset_ in **[Describing Textures in the Wild (Cimpoi et al., 2014)](https://arxiv.org/abs/1311.3618)** -- **DTD** from now on.  
  In `/data` you can find the texture images reported in the paper (`blotchy.jpg`, `matted.png`, `scaly.png`, `striped.png`) and the image `pebbles.jpg` from **[Texture synthesis using convolutional neural networks (Gatys et al., 2015)](https://arxiv.org/abs/1505.07376)**.
- **Models**
  All models are available on Torchvision:
    ```text
  Alexnet = alexnet-owt-7be5be79.pth
  Densenet-121 = densenet121-a639ec97.pth
  Densenet-169 = densenet169-b2777c0a.pth
  Densenet-201 = densenet201-c 1103571.pth
  Inception-v3 = inception_v3_google-1a9a5a14.pth
  Mobilenet = mobilenet_v2-b0353104.pth
  Resnet18 = resnet18-5c106cde.pth
  Resnet34 = resnet34-333f7ec4.pth
  Resnet50 = resnet50-19c8e357.pth
  Resnet101 = resnet101-5d3b4d8f.pth
  Resnet152 = resnet152-b121ed2d.pth
  VGG16 = vgg16_bn-6c64b313.pth
  VGG19 = vgg19_bn-c79401a0.pth
- **Image optimization and feature extraction**  
  Synthesize one texture sample from the images in `/data` by running `/src/image_optimization.py`.  
  Extract features from the images with one forward pass running `/src/extract_gram_representations.py`.  

- **Analyses**  
   The following scripts work on the features extracted as per the previous point.  
  `/src/rsa_gram.py` performs Representational Similarity Analysis.  
  `/src/mutual_info_estimate.py` computes Mutual Information.  
  `/src/brainscore_correlations.py` performs correlation against **[BrainScore (Schrimpf et al., 2019)](https://arxiv.org/abs/1909.06161)**.  
  Notes: `/analyses/dataloader_dtd.py` and `/analyses/dataloader_gaussian.py` contain two dataloaders as a standard pipeline to deal with data in PyTorch and Torchvision.  
