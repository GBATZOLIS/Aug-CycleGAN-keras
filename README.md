# MULTI MODAL IMAGE ENHANCEMENT

Our work has been based on the augmented Cycle-GAN framework. Crucial modifications were maded on the framework before it was trained on the DPED dataset to achieve multimodal/diverse image enhancement. 

Below you can see some visual results. The image on the left is the low quality image, while the gif on the right shows different enhancements of the low quality image.


![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/790.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/833.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/849.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/936.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1034.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1176.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1428.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1443.gif)

The following gifs are relevant to the mixed regularisation protocol which I devised in order to control the diversity-quality trade-off. The mixed regularisation protocol is explained in my master thesis report. Sample visual results after the first stage of the training are shown below. The left image is the LQ image, the gif in the middle contains random samples from the modelled conditional distribution P(HQ|LQ) and the image on the right is the ground truth HQ image. The enhanced images contain random color-shifts which are caused by mode seeking regularisation.

![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/168_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/220_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/833_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/847_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/1121_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/1443_opt.gif)

During the second stage of the training, I use only PPL regularisation to make the artefacts created by mode seeking regularisation in the first stage disappear. PPL regularisation removes the artefacts at a small compromise on diversity. Visual results are shown below.

![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/168_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/220_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/833_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/847_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/1121_opt.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/1443_opt.gif)








