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

## Stack

- **React** - For building a Progressive Web App for users which provides a mobile application experience using web apps.
- **Firebase** - For using as backend for features of application other than image analysis.
- **Flask (Python)** - For building a single endpoint to expose Machine Learning Model to predict quality of image.


### Requirement Gathering
The Google doc for the Explore ML Mentorship Bootcamp 
[**doc**](https://docs.google.com/document/d/1ze_USE1D-ME89PkI3o_sCu7ftZMBVtGC6GwikKJPYLo/edit)

