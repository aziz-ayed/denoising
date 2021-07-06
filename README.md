# Deep denoising for the MCCD PSF model

Research aimed at integrating Deep Learning denoising into PSF modelling packages.

---
> Main contributors: <a href="https://github.com/aziz-ayed" target="_blank" style="text-decoration:none; color: #F08080">Aziz Ayed</a> - <a href="https://tobias-liaudat.github.io" target="_blank" style="text-decoration:none; color: #F08080">Tobias Liaudat</a>  
> Email: <a href="mailto:aziz.ayed@hec.edu" style="text-decoration:none; color: #F08080">aziz.ayed@hec.edu</a> - <a href="mailto:tobias.liaudat@cea.fr" style="text-decoration:none; color: #F08080">tobias.liaudat@cea.fr</a>  
> Documentation:   
> Article:  
> Current release: 06/07/2021
---

Point Spread Function modelling usually does not include a denoising step, as this often induces some bias in shape reconstruction. In fact, in the weak lensing regime, we would rather have noisy but errorless shapewise reconstructions than noiseless reconstructions with some error in shape measurement, no matter how small.  
In this repository, we record our research aimed at introducing Deep denoising into the non-parametric MCCD PSF modelling (<a href="https://arxiv.org/pdf/2011.09835.pdf" target="_blank" style="text-decoration:none; color: #F08080">paper</a>) with the goal of increasing the precision of the algorithm without introducing error in shape reconstruction.  
  
Our version of the MCCD algorithm with the integrated neural networks can be found <a href="https://github.com/aziz-ayed/mccd.git" target="_blank" style="text-decoration:none; color: #F08080">here</a>.

1. [Packages Used](#packages-used)
1. [Datasets](#datasets)
1. [Models](#models)
1. [Results](#results)


## Packages Used

- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [astropy](https://github.com/astropy/astropy)
- [GalSim](https://github.com/GalSim-developers/GalSim)
- [ModOpt](https://github.com/CEA-COSMIC/ModOpt)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [Original MCCD](https://github.com/CosmoStat/mccd)

## Datasets

We started by training our algorithm with simulated stars catalogs created with the GalSim package. For the sake of storage space, we did not upload our datasets in this repository, but the code used to generate them can be found in the <a href="https://github.com/aziz-ayed/denoising/blob/main/dataset_generation_v2.ipynb" target="_blank" style="text-decoration:none; color: #F08080">dataset generation</a> notebook.  
Stars are parametrized by 2 ellipticity components ``e1`` ``e2`` and one size component ``R2``.  
The datasets used can thus be reproduced using the notebook above and the parameters catalogs specified in the <a href="https://github.com/aziz-ayed/denoising/issues" target="_blank" style="text-decoration:none; color: #F08080">Issues</a> section of this repository, for each step of the project.  
To generate our datasets, we preprocess clean simulated images with Gaussian noise. Thus, we produced a study of CFIS data to estimate the real-life noise levels - expressed in terms of Signal to Noise ratio - than can be found in the <a href="https://github.com/aziz-ayed/denoising/tree/main/SNR_study" target="_blank" style="text-decoration:none; color: #F08080">SNR study</a> folder.
  
However, the MCCD algorithm operates on eigenPSFs and not simple stars, which led us to designing a more complex training dataset. Our methodology and notebooks are respectively detailled in the dedicated <a href="https://github.com/aziz-ayed/denoising/blob/main/datasets/README.md" target="_blank" style="text-decoration:none; color: #F08080">README</a> and the <a href="https://github.com/aziz-ayed/denoising/tree/main/datasets" target="_blank" style="text-decoration:none; color: #F08080">datasets</a> folder. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/57011275/124650455-10049e00-de9a-11eb-8e64-0e1da3869e5b.jpg" alt="star_vs_eigenpsf" width="50%" height="50%"/>
</p>


## Models

For our project, we decided to work with two types of neural networks. Both models are implemented in TensorFlow and the code can be found in the ``unets`` and ``learnlets`` folders. The trained models with best performances when implemented in the MCCD algorithm can be found in the <a href="https://github.com/aziz-ayed/denoising/tree/main/Trained%20models/best_models" target="_blank" style="text-decoration:none; color: #F08080">best models</a> folder.

### U-Nets

U-Nets are Deep Neural Networks widely used in vision tasks, and are considered to be state-of-the-art when it comes to image denoising. Primarly designed for Biomedical Image Segmentation, they have an architecture that consists of ``a contracting path to capture context and a symmetric expanding path that enables precise localization``, giving it its characteristical U shape (<a href="https://arxiv.org/pdf/1505.04597.pdf" target="_blank" style="text-decoration:none; color: #F08080">paper</a>).  
Throughout the project, we worked with different architectures that are detailled in the Issues section of this repository, and the final network is composed of ``16 base filters``, and the training is done in the <a href="https://github.com/aziz-ayed/denoising/blob/main/unets_training.ipynb" target="_blank" style="text-decoration:none; color: #F08080">unets_training.ipynb</a> notebook.

<p align="center">
  <img src="https://user-images.githubusercontent.com/57011275/124658869-99b96900-dea4-11eb-9c7b-101d4956c5de.jpg" alt="star_vs_eigenpsf" width="50%" height="50%"/>
</p>

### Learnlets

Learnlets are based on a network architecture that ``conserves the properties of sparsity based methods such as exact reconstruction and good generalization properties, while fostering the power of neural networks for learning and fast calculation`` (<a href="https://hal.archives-ouvertes.fr/hal-03020214/document" target="_blank" style="text-decoration:none; color: #F08080">paper</a>).  
Our idea is to leverage the properties of sparsity based methods to ensure solid shape reconstruction and reduce the mismeasurement introduced by our denoising algorithm.  
Throughout the project, we worked with different architectures that are detailled in the Issues section of this repository, and the final network is composed of ``256 filters``. We started with a simple version of the Learnlets that can be found in the <a href="https://github.com/aziz-ayed/denoising/blob/main/learnlets_original_training.ipynb" target="_blank" style="text-decoration:none; color: #F08080">learnlets_original_training.ipynb</a> notebook. However, our final network uses dynamic thresholding with an estimation of the noise level of the image as an input, which can be found in the <a href="https://github.com/aziz-ayed/denoising/blob/main/learnlets_dynamic_training.ipynb" target="_blank" style="text-decoration:none; color: #F08080">learnlets_dynamic_training.ipynb</a> notebook.

<p align="center">
  <img src="https://user-images.githubusercontent.com/57011275/124666235-09802180-deae-11eb-89c0-297b23d64ce5.jpg" width="50%" height="50%"/>
</p> 


## Results

The progressive results of the standalone denoising at each step of our project can be found in the <a href="https://github.com/aziz-ayed/denoising/issues" target="_blank" style="text-decoration:none; color: #F08080">Issues</a> section of this repository.  
The following table presents the final results of the different methods trained on the complex dataset (60% of eigenPSFs and 40% of simulated stars) when integrated into MCCD.  

<p align="center">
  <img src="https://user-images.githubusercontent.com/57011275/124672536-7a780700-deb7-11eb-875e-404823d6fc81.jpg" width="70%" height="70%"/>
</p>

More detailled results can be found in the <a href="https://github.com/aziz-ayed/denoising/tree/main/results" target="_blank" style="text-decoration:none; color: #F08080">dedicated</a> folder and <a href="https://github.com/aziz-ayed/denoising/blob/main/results/README.md" target="_blank" style="text-decoration:none; color: #F08080">README</a>.  
Some complementary results are also presented in the <a href="https://github.com/aziz-ayed/mccd/issues" target="_blank" style="text-decoration:none; color: #F08080">Issues</a> section of our fork of MCCD.

