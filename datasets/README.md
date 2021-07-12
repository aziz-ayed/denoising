# Datasets

As stated in the main README:

> We started by training our algorithm with simulated stars catalogs created with the GalSim package. For the sake of storage space, we did not upload our datasets in this repository, but the code used to generate them can be found in the <a href="https://github.com/aziz-ayed/denoising/blob/main/dataset_generation_v2.ipynb" target="_blank" style="text-decoration:none; color: #F08080">dataset generation</a> notebook.  
Stars are parametrized by 2 ellipticity components ``e1`` ``e2`` and one size component ``R2``.  
The datasets used can thus be reproduced using the notebook above and the parameters catalogs specified in the <a href="https://github.com/aziz-ayed/denoising/issues" target="_blank" style="text-decoration:none; color: #F08080">Issues</a> section of this repository, for each step of the project.  
To generate our datasets, we preprocess clean simulated images with Gaussian noise. Thus, we produced a study of CFIS data to estimate the real-life noise levels - expressed in terms of Signal to Noise ratio - than can be found in the <a href="https://github.com/aziz-ayed/denoising/tree/main/SNR_study" target="_blank" style="text-decoration:none; color: #F08080">SNR study</a> folder.
  
> However, the MCCD algorithm operates on eigenPSFs and not simple stars, which led us to designing a more complex training dataset.

Thus, we designed two more complex training sets, the <a href="https://github.com/aziz-ayed/denoising/tree/main/datasets/polynomial_dataset" target="_blank" style="text-decoration:none; color: #F08080">polynomial dataset</a> and the <a href="https://github.com/aziz-ayed/denoising/tree/main/datasets/realistic_dataset" target="_blank" style="text-decoration:none; color: #F08080">realistic dataset</a>.

## Polynomial dataset

To train our model, we needed to generate a catalog of eigenPSFs. The most straight-foward solution was to generate exposures of simulated stars without noise, pass them through the MCCD algorithm without any denoising, and retrieve the fitted eigenPSFs.  
However, to do so, we needed to build exposures of simulated stars that were close to real-world data, otherwise our training eigenPSFs would have been too different from the ones our model will have to deal with when put in production.  
Thus, the first step was to generate a set of functions that take in input a given star position and compute the ellipticity (e1 and e2) and the size (R2) of the star. We decided to work with polynomial functions to reproduce the smooth variations of real-life data, and our methodology can be found in the <a href="https://github.com/aziz-ayed/denoising/blob/main/datasets/polynomial_dataset/generate_dicts_poly.ipynb" target="_blank" style="text-decoration:none; color: #F08080">generate_dicts_poly notebook</a>.

<p align="center">
  <img src="https://user-images.githubusercontent.com/57011275/125307963-cfd96b80-e330-11eb-9815-a2d3eb6929c4.png" alt="Polynomial variations" width="200%" height="200%"/>
</p>

Using different sets of three different functions for each exposure, we generated 20 exposures of stars thanks to the Galsim package and a custom GenerateSimDataset function that can be found in the <a href="https://github.com/aziz-ayed/denoising/blob/main/datasets/polynomial_dataset/generate_stars_catalog.ipynb" target="_blank" style="text-decoration:none; color: #F08080">generate_stars_catalog notebook</a>.  
For example, we plot hereafter the parameters' variations in one of the twenty exposures. 

e1 variations    |  e2 variations     |  R2 variations
:-------------------:|:------------------: |:-----------------:
![e1](https://user-images.githubusercontent.com/57011275/125311327-9f470100-e333-11eb-9958-c921f1779569.png) | ![e2](https://user-images.githubusercontent.com/57011275/125311393-aa9a2c80-e333-11eb-9478-4c01b33888af.png) | ![R2](https://user-images.githubusercontent.com/57011275/125311554-c7cefb00-e333-11eb-8dde-1689d735deb4.png)

Then, we ran MCCD on each of the exposures and extracted 605 eigenPSFs (45 global and 560 local) fitted model (one per exposure), that we preprocessed with Gaussian noise. To these 12 100 eigenPSFs, we added around 8 500 simulated regular stars, in order to obtain a complete dataset with a 60/40 repartition between eigenPSFs and simulated stars that we used to train our model.   
Our methodology can be found in the <a href="https://github.com/aziz-ayed/denoising/blob/main/datasets/polynomial_dataset/extracting_eigenpsfs.ipynb" target="_blank" style="text-decoration:none; color: #F08080">extracting_eigenpsfs notebook</a>.  

<p align="center">
  <img src="https://user-images.githubusercontent.com/57011275/125316251-3d3cca80-e338-11eb-95da-9bea03599f65.png" alt="Final polynomial dataset" width="100%" height="100%"/>
</p>

## Realistic dataset



