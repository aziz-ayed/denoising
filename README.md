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
In this repository, we record our research aimed at introducing Deep denoising into the non-parametric MCCD PSF modelling with the goal of increasing the precision of the algorithm without introducing error in shape reconstruction.  
  
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
  
However, the MCCD algorithm operates on eigenPSFs and not simple stars, which led us to designing a more complex training dataset. Our methodology and notebooks are respectively detailled in the dedicated <a href="https://github.com/aziz-ayed/denoising/blob/main/datasets/README.md" target="_blank" style="text-decoration:none; color: #F08080">README</a> and <a href="https://github.com/aziz-ayed/denoising/tree/main/datasets" target="_blank" style="text-decoration:none; color: #F08080">datasets</a> folder. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/57011275/124650455-10049e00-de9a-11eb-8e64-0e1da3869e5b.jpg" alt="star_vs_eigenpsf" width="50%" height="50%"/>
</p>


## Models



## Results


