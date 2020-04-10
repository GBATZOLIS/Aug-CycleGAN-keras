# Improved Augmented CycleGAN model.

It has been modified to perform optimally in the task of multimodal image enhancement and super-resolution. The modifications have been inspired from the state-of-the-art StyleGAN2 model.

The model has been trained in a semi-supervised manner on the DPED smartpohone dataset.
During testing phase, the model receives a low quality image that we want to enhance and a latent vector. The model output is an enhanced version of the low quality image. By varying the latent code, we vary the output enhanced image. Essentially, the model lets us explore the manifold of high quality images which correspond to a specific low quality image. This approach is very interesting as it tackles this ill-posed problem in a probabilistic manner and it lets us perform perceptuall studies which can increases our capacity to describe the human visual system.

Below, we present many gif examples which show the performance of the model. 
On the left you can see the low quality image, in the middle the varying output enhanced high quality image and on the right the ground truth high quality image


