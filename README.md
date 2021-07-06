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

1. [Dependencies](#packages-used)
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

## Models

## Results


