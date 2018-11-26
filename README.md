# BiSeNet-Implementation
This repo contains the code for a fierce attempt to implement this amazing Research paper using Keras.


Description
-
In this repo you will find the all steps necessary to build a Semantic segmentation model.
You will learn how to do the following:
    Reading images
    Image Preprocessing
    Data pipeline
    Building the Model

Any suggestions to improve this repository, including any new segmentation models you would like to see are welcome!

Citing
-
If you find this repository useful, please consider citing it using a link to the repo. :)

Reading Images
-
The [getImages](https://github.com/Blaizzy/BiSeNet-Implementation/blob/master/getImages.ipynb) notebook has a step by step how to read images from any directory. For a more detailed guide you can check my article on medium [BiSeNet for Real-Time Segmentation Part II](https://medium.com/@prince.canuma/bisenet-for-real-time-segmentation-part-ii-32e189a4aed5).

Preprocessing
-
The [Preprocessing](https://github.com/Blaizzy/BiSeNet-Implementation/blob/master/Preprocessing.ipynb) notebook has a full image preprocessing tutorial using opencv and includes getting segmentation masks. For a more detailed and personalized guide you can check my article on medium [Image Pre-processing](https://towardsdatascience.com/image-pre-processing-c1aec0be3edf).

Model
-
The Model notebook is still under development.
Based on the Bilateral Segmentation Network paper on arxiv:
https://arxiv.org/abs/1808.00897

So far, only the BiSeNet model is done.
Difficulties are: 

...* Auxiliary Loss function(softmax). 
