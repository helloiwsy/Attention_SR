Attention 모델을 이용한 단일 영상 초고해상도 복원 기술 in 2020 한국방송미디어공학회 하계학술대회
-----
## 0. Please Note This
This repositorty is a tensorflow implementations the paper.
Since the code is writeen for study,this may include incorrect implementation or error.


## 1. Prepare dataset for training
  - This network require (I_LR, HF_HR, LF_HR) unlike existing normal deep learning based super resolution network.
    - I_LR : Low-resolution Input image
    - HF_HR : High-resolution High-Frequency image of Original High resolution image
    - LF_HR : High-resolution Low-Frequency image of Original High resolution image
  1. Use your datset such as DIV2K, ImageNet .... 
  2. Given a high original high resolution image I_HR, Make low-resolution input image I_LR using interpolation or downscale method
  3. By applyging [SATV](https://github.com/decpearl/satv), Make LF_HR and HF_HR
  4. Save the image, (Original HR, HF_HR, LF_HF, I_LR) as( xxx_HR.mat,xxx_HRHF.mat, xxx_HRLF.mat xxx_LR.mat ) with ['HR']['HRHF']['HRLF']['LR'], respectively.
  5. Locate the files like below
  - ./
    - train_HF
      - 001_HR.mat
      - 001_HRHF.mat
      - 001_HRLF.mat
      - 002_HR.mat
      - 002_HRHF.mat
      - 002_HRLF.mat
      - 003_HR.mat
      - 003_HRHF.mat
      - 003_HRLF.mat
      - ...
    - train_LR
      - 001_LR.mat
      - 002_LR.mat
      - 003_LR.mat      
      - ...
  6. When you run train.py, pass the train_HR path like '/yourpath/trian_HR/*_HR.mat'
  
## 2. Train Model
  - In DSR.py
    - you can choice attention module or make attention model yourself.
  - run train.py
    - python train.py --hr_path='/your_path/train_HR/*_HR.mat'
      (I recommend edit the path in code, do not type in teminal)
  
## 3. Test 
  - If you finish training, Oepn test.py and type your test file path


---

# Requirements
- tensorflow 1.14

---
The code is based on the [code](https://github.com/geonm/EnhanceNet-Tensorflow.git), thanks for the author.

    
