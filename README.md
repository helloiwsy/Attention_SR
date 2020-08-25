# Attention_SR
-----
Tensorflow  Implementation of Astudy on Single Image Super Resolution Using Attention Model(Attention 모델을 이용한 단일 영상 초고해상도 복원 기술) in 2020 한국방송미디어공학회 하계학술대회
-----

## Prepare dataset for training
  - This network require (I_LR, HF_HR, LF_HR) unlike existing normal deep learning based super resolution network.
    - I_LR : Low-resolution Input image
    - HF_HR : High-resolution High-Frequency image of Original High resolution image
    - LF_HR : High-resolution Low-Frequency image of Original High resolution image
  1. Use your datset such as DIV2K, ImageNet .... 
  2. Given a high original high resolution image I_HR, Make low-resolution input image I_LR using interpolation or downscale method
  3. By applyging [SATV](https://github.com/decpearl/satv), Make LF_HR and HF_HR
  4. Save the image, (Original HR, HF_HR, LF_HF, I_LR) as( xxx_HR.mat,xxx_HRHF.mat, xxx_HRLF.mat xxx_LR.mat )
  5. Locate the files like below

-train_HF
-001_HR.mat
-001_HRHF.mat
-001_HRLF.mat

