# Improved Augmented CycleGAN model.

The original paper has been modified to perform optimally in the task of multimodal image enhancement and super-resolution. The modifications have been inspired from the state-of-the-art StyleGAN2 model. The main modification is the replacement of the Conditional Instance Normalisation layer by the Conditional Modulated Convolutional layer (this layer has been in introduced in the StyleGAN 2 model), which eliminated the blob-like artefacts which are not only apparent in my previous trainings, but also in the official results of the Augmented CycleGAN paper. Another modification, which helps improve performance is the blur loss between the input and output of the image generators, which motivates the generators to preserve the low frequency components and essentially focus more on the generation of the high frequency components.

The model has been trained in a semi-supervised manner on the DPED smartpohone dataset.
During testing phase, the model receives as input a low quality image (that we want to enhance) and a latent vector. The model output is an enhanced version of the low quality image. By varying the latent code, we vary the output enhanced image. Essentially, the model lets us explore the manifold of high quality images which correspond to a specific low quality image. This approach is very interesting as it tackles this ill-posed problem in a stochastic manner and it lets us perform perceptual studies which can explore the human visual system.

Below, we present gif examples which show how the model performs on the test part of the DPED dataset.

On the left you can see the low quality image, in the middle the varying output enhanced high quality image and on the right the ground truth high quality image. In the following gifs, the output images are sampled from the modelled conditional distribution P(Y|X) where X is the given low quality image and Y is the enhanced image.


![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/1047.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/1628.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/1443.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/1428.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/1701.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/484.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/554.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/833.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/97.gif)





