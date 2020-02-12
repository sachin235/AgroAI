# AgroAI

Project to build an unbiased platform for farmers to predict quality and price of the crops.

## Problem

In India over 70 percent of the rural household depend on Agriculture. It is important to give the farmers the right value of their produce depending on the quality. It has been observed that many farmers don’t even get the Minimum Support Price (MSP) for their produce due to the ill practices used by middlemen who are continuously widening the gap between the pay to farmers and the price taken from the customer.

## Idea

- The main idea is to build an application which will use the camera of mobile to determine the crop quality which can be used to predict the value of crops.
- Computer Vision (Machine Learning/Artificial Intelligence) will be used to determine the quality of the crop using the different visible features of the crop.
- This can enable farmers to get the right price of their crops based on their quality without any middlemen.
- Traders (middlemen) can buy crops on the platform only at prices determined by the model. This will ensure that farmers are not exploited by the traders.
- Building such a platform will enable the government to analyze the data from different farmers across the country and advice farmers accordingly.
- The practices used by the farmers producing good quality crops can be shared with others by integrating a platform like “StackOverflow” where farmers/government can post about the good practices.

## Implementation Plan

- Determine quality of crop/produce using the image of crop.
- Building application with features like question addition, deletion and updation, answering question, image uploading.
- Integrate quality determination model to platform.
- (Optional) Platform for analysis of gathered data.

## Dataset Description

The dataset consists of images of wheat grains organised into five folders - grain, damaged_grain, foreign, broken_grain, grain_cover.

- **Grain** - contains images of healthy wheat grains

  <p align="center">
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain/IMG_20161016_122456328_1.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain/IMG_20161016_122456328_10.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain/IMG_20161016_122456328_103.jpg?raw=true" width = 100 height = 100>
  </p>

- **Damaged_grain** - contains images of non-healthy or deformed wheat grains

  <p align="center">
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/damaged_grain/IMG_20161016_124705064_395.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/damaged_grain/IMG_20161016_124705064_401.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/damaged_grain/IMG_20161016_124705064_416.jpg?raw=true" width = 100 height = 100>
  </p>

- **Foreign** - contains images of particles other than wheat grains

  <p align="center">
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/foreign_particles/IMG_20161016_125744060_5734.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/foreign_particles/IMG_20161016_125744060_5759.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/foreign_particles/IMG_20161016_125744060_5756.jpg?raw=true" width = 100 height = 100>
  </p>

- **Broken_grain** - contains images of broken wheat grains

  <p align="center">
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain_broken/IMG_20161016_124705064_406.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain_broken/IMG_20161016_124705064_430.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain_broken/IMG_20161016_124705064_432.jpg?raw=true" width = 100 height = 100>
  </p>

- **Grain_cover** - contains images of cover of wheat grains
  <p align="center">
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain_covered/IMG_20161016_131740_4803.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain_covered/IMG_20161016_131740_4806.jpg?raw=true" width = 100 height = 100>
    <img src = "https://github.com/sachin235/AgroAI/blob/master/Dataset/grain_covered/IMG_20161016_131740_4816.jpg?raw=true" width = 100 height = 100>
  </p>

Source: https://github.com/deepakrana47/Wheat-quality-detector-2/tree/master/dataset5_dep_on_4

## Running Model

1. Create Virtual env - Use [link](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to install env in python3

2. Install dependencies - `pip install -r "requirements.txt"`

3. Run the code - `python "wheat_quality_predictor.py" "test3.jpg"`

## Stack

- **React** - For building a Progressive Web App for users which provides a mobile application experience using web apps.
- **Node + Mongo** - For using as backend for features of application other than image analysis.
- **Flask (Python)** - For building a single endpoint to expose Machine Learning Model to predict quality of image.

## Requirement Gathering

The Google
[**doc**](https://docs.google.com/document/d/1ze_USE1D-ME89PkI3o_sCu7ftZMBVtGC6GwikKJPYLo/edit)
for the Explore ML Mentorship Bootcamp.

## References

- [**Machine Learning for prediction of Crop Yield**](https://medium.com/ai-techsystems/prediction-of-crop-yield-using-machine-learning-84fcd9e0649a)
- [**Research Paper on Prediction of Crop yeild using ML**](https://www.irjet.net/archives/V5/i2/IRJET-V5I2479.pdf)
- [**Crop Yield Prediction Using Deep Neural Networks**](https://www.frontiersin.org/articles/10.3389/fpls.2019.00621/full)
- [**An app that helps farmers cut the middleman out**](https://www.livemint.com/Consumer/nQLEyDHTQvkVAodbfA6B9L/An-app-that-helps-farmers-cut-the-middleman-out.html)
- [**Farmers, Middlemen and the Way Out**](https://spontaneousorder.in/farmers-middlemen-and-the-way-out/)
- [**Lean Supply Chain**](https://www.thebetterindia.com/52355/empowering-farmers-greenagtech/)
