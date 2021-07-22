#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import itertools
import mccd


## Extracting the EigenPSFs from the fitted models

vignets_noiseless = np.zeros((19665, 51, 51))

i=0

for j in list(range(2000000, 2000287)) + list(range(2100000, 2100150)):
    
    path_fitted_model = '/n05data/ayed/outputs/mccd_runs/shapepipe_run_2021-07-13_11-19-06/mccd_fit_val_runner/output/fitted_model-'+str(j)+'.npy'
    fitted_model = np.load(path_fitted_model, allow_pickle=True)
    S = fitted_model[1]['S']

    S_loc = S[-1]
    vignets_noiseless[i*45:(i+1)*45, :, :] = mccd.utils.reg_format(S_loc)
    i+=1

        

np.random.shuffle(vignets_noiseless)
        

## Saving the dataset

train_dic = {'VIGNETS_NOISELESS': vignets_noiseless}

mccd.mccd_utils.save_to_fits(train_dic, '/n05data/ayed/outputs/eigenpsfs/global_eigenpsfs.fits')

    


# In[ ]:




