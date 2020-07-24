# Image Inpainting using Partial Convolutions on Google Colaboratory
This is the Google Colaboratory implementation of [Liu et al., 2018. Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723).

![Example_01](result_images/result_01.png?raw=true "Example_01")  
![Example_02](result_images/result_02.png?raw=true "Example_02")  
![Example_03](result_images/result_03.png?raw=true "Example_03")


Many ideas used here were came from [this](https://github.com/MathiasGruber/PConv-Keras) & [this](https://github.com/ezavarygin/PConv2D_Keras) brilliant repositories.  
Differences between theirs and mine:  
  - **Support for Tensorflow 2.2.0.**  
    All works fine on Google Colaboratory as of July 2020.  
  - **No datasets are necessary.**  
    The dataset we are using is a filtered version of <a href="https://www.kaggle.com/c/dogs-vs-cats/data" target="_blank">Dogs vs. Cats</a> dataset from Kaggle (provided by Microsoft Research).  
    The dataset is loaded <a href="https://www.kaggle.com/c/dogs-vs-cats/data" target="_blank">this way</a>. The dataset has following directory structure.    
    <pre style="font-size: 10.0pt; font-family: Arial; line-height: 2; letter-spacing: 1.0pt;" >
    <b>cats_and_dogs_filtered: 3000</b>  
    |__ <b>train: 2000</b>  
        |______ <b>cats</b>: [cat.0.jpg, cat.1.jpg, cat.2.jpg ....]  
        |______ <b>dogs</b>: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]  
    |__ <b>validation: 1000</b>  
        |______ <b>cats</b>: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ....]  
        |______ <b>dogs</b>: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]  
    </pre>  

    Of course, you can use your own (and larger) dataset. The filtered dataset may be small to get the same result as demonstrated in the paper. I just put a premium on making this repo Colab-friendly.  
    The cute cats & dogs images in the `dataset/test` directory were downloaded from ImageNet.  

## Requirements
- Nothing to run on Google Colaboratory.
  The following libraries we use in the Colab are installed by default as of July 2020.  
    - Python 3.6.9
    - TensorFlow 2.2.0
    - PyTorch 1.5.1+cu101
    - OpenCV 4.1.2
    - NumPy 1.18.5

## How to run
1. Upload whole directory (`PConv-Tensorflow2`) to your Google Drive.  
2. Open `image-inpainting.ipynb` with Google Colabratory.  
3. Enable GPUs for the notebook:  
    - Navigate to `Edit`→`Notebook` Settings, then  
    - Select `GPU` from the Hardware Accelerator drop-down.  
4. Run each cell step by step.  

## Training
The model was trained in two steps:  
1. **Initial training**
  - The Batch Normalization parameters enabled.
  - 30 epochs with a learning rate of 0.0002.  
2. **Fine-tuning**  
  - The Batch Normalization parameters freezed in the encoder part of the network.
  - 8 epochs with a learning rate of 0.00005.

![Example_04](result_images/result_04.png?raw=true "Example_04")

## Comments
Many thanks to [@MathiasGruber](https://github.com/MathiasGruber) and [@ezavarygin](https://github.com/ezavarygin) for sharing their partial convolution implementation.  
This is my first deep learning & Tensorflow experience. Let me know if there is anything that needs improvement.  
