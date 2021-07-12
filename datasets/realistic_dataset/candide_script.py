#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import mccd
import galsim as gs
import mccd.mccd_utils as mccd_utils
import mccd.dataset_generation as dataset_generation
import mccd.utils as utils
import scipy as sp
import argparse


# In[3]:


parser = argparse.ArgumentParser()

parser.add_argument('--catalog_bin', type=int, default=0)

args = parser.parse_args()

e1_path = '/n05data/ayed/data/moments/e1_psf.npy'
e2_path = '/n05data/ayed/data/moments/e2_psf.npy'
fwhm_path = '/n05data/ayed/data/moments/seeing_distribution.npy'
output_path = '/n05data/ayed/outputs/datasets'

for i in [2000000 + i + 34*args.catalog_bin for i in range (34)]:
    
    catalog_id = i
    sim_dataset_generator = dataset_generation.GenerateRealisticDataset(e1_path=e1_path, e2_path=e2_path, size_path=fwhm_path, output_path=output_path, catalog_id=catalog_id)
    sim_dataset_generator.generate_train_data()
    sim_dataset_generator.generate_test_data(x_grid=5, y_grid=10)
    
    