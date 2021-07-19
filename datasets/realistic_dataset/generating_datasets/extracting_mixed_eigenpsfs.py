#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import itertools
import mccd


## Extracting the EigenPSFs from the fitted models

vignets_noiseless = np.zeros((264385, 51, 51))


i=0

for j in list(range(2000000, 2000287)) + list(range(2100000, 2100150)):
    
    path_fitted_model = '/n05data/ayed/outputs/mccd_runs/shapepipe_run_2021-07-13_11-19-06/mccd_fit_val_runner/output/fitted_model-'+str(j)+'.npy'
    fitted_model = np.load(path_fitted_model, allow_pickle=True)
    S = fitted_model[1]['S']

     for k in range (40):
        S_loc = S[k]
        vignets_noiseless[i*605+k*14:i*605+(k+1)*14, :, :] = mccd.utils.reg_format(S_loc)
     
    vignets_noiseless[i*605+40*14:i*605+40*14+45, :, :] = mccd.utils.reg_format(S[-1])
        
    i+=1
        

np.random.shuffle(vignets_noiseless)
        
#for i in range (len(vignets_noiseless)):
#    vignets_noiseless[i] = vignets_noiseless[i]/np.linalg.norm(vignets_noiseless[i])



train_dic = {'VIGNETS_NOISELESS': vignets_noiseless}

mccd.mccd_utils.save_to_fits(train_dic, '/n05data/ayed/outputs/eigenpsfs/dataset_eigenpsfs.fits')

    


# In[ ]:




