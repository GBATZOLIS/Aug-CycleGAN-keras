# MULTI MODAL IMAGE ENHANCEMENT

The Augmented CycleGAN framework was modified and trained on the DPED dataset to achieve multimodal/diverse image enhancement. 

Below you can see visual results. The image on the left is the low quality image, while the gif on the right shows different enhanced versions of the low quality image.


![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/790.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/833.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/849.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/936.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1034.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1176.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1428.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1443.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_only_gifs/1494.gif)

The following gifs are relevant to the evaluation protocol which I devised and is explained in my master thesis report. Sample visual results after the first stage of the training are shown below. The enhanced images contain random color-shifts which are created by mode seeking regularisation.

![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/168.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/220.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/833.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/847.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/1121.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/Mixed_Gifs/1443.gif)

During the second stage of the training, I use only PPL regularisation to make the artefacts created by mode seeking regularisation in the first stage disappear. Visual results are shown below.

![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/168.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/220.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/833.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/847.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/1121.gif)
![Alt Text](https://github.com/GBATZOLIS/Aug-CycleGAN-keras/blob/master/progress/gif/PPL_gifs/1443.gif)








