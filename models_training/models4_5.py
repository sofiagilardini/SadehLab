import numpy as np 
import matplotlib.pyplot as plt
import os
import fnmatch
import cebra
import functions as aux_f
import torch
import classes
import random 
import cebramodels as models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import functions4 as aux_f4
from cebra import CEBRA
from copy import deepcopy

# all the same except batch size doubled to 1024 -- tried this not successful - I am now trying to use the same batch size as the original model, but with output dimensions = 6
# saved in models 5, 6, 7, 8


# set seed
SEED = 49
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# define device
if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Using device: ', device)

# testing KNN 



# load indices for training and testing - not stationary filtered

s3_indices_tr = np.load('./information/indices/A118_s3_indices_tr.npy', allow_pickle=True)
s3_indices_test = np.load('./information/indices/A118_s3_indices_test.npy', allow_pickle=True)
s4_indices_tr = np.load('./information/indices/A118_s4_indices_tr.npy', allow_pickle=True)
s4_indices_test = np.load('./information/indices/A118_s4_indices_test.npy', allow_pickle=True)
s5_indices_tr = np.load('./information/indices/A118_s5_indices_tr.npy', allow_pickle=True)
s5_indices_test = np.load('./information/indices/A118_s5_indices_test.npy', allow_pickle=True)

# load data 

s3_neuraldata_tr = np.load('./information/neuraldata/A118_neuraldata_session3_tr.npy', allow_pickle=True)
s3_neuraldata_test = np.load('./information/neuraldata/A118_neuraldata_session3_test.npy', allow_pickle=True)

s4_neuraldata_tr = np.load('./information/neuraldata/A118_neuraldata_session4_tr.npy', allow_pickle=True)
s4_neuraldata_test = np.load('./information/neuraldata/A118_neuraldata_session4_test.npy', allow_pickle=True)

s5_neuraldata_tr = np.load('./information/neuraldata/A118_neuraldata_session5_tr.npy', allow_pickle=True)
s5_neuraldata_test = np.load('./information/neuraldata/A118_neuraldata_session5_test.npy', allow_pickle=True)

# load the filmID data

s3_filmID_tr = np.load('./information/filmID/A118_s3_filmID_tr.npy', allow_pickle=True)
s3_filmID_test = np.load('./information/filmID/A118_s3_filmID_test.npy', allow_pickle=True)

s4_filmID_tr = np.load('./information/filmID/A118_s4_filmID_tr.npy', allow_pickle=True)
s4_filmID_test = np.load('./information/filmID/A118_s4_filmID_test.npy', allow_pickle=True)

s5_filmID_tr = np.load('./information/filmID/A118_s5_filmID_tr.npy', allow_pickle=True)
s5_filmID_test = np.load('./information/filmID/A118_s5_filmID_test.npy', allow_pickle=True)


# load the pupildiam data 

s3_pupilDiam_tr = np.load('./information/pupilDiam/A118_s3_pupilDiam_tr.npy', allow_pickle=True)
s3_pupilDiam_test = np.load('./information/pupilDiam/A118_s3_pupilDiam_test.npy', allow_pickle=True)
s4_pupilDiam_tr = np.load('./information/pupilDiam/A118_s4_pupilDiam_tr.npy', allow_pickle=True)
s4_pupilDiam_test = np.load('./information/pupilDiam/A118_s4_pupilDiam_test.npy', allow_pickle=True)
s5_pupilDiam_tr = np.load('./information/pupilDiam/A118_s5_pupilDiam_tr.npy', allow_pickle=True)
s5_pupilDiam_test = np.load('./information/pupilDiam/A118_s5_pupilDiam_test.npy', allow_pickle=True)

# make nan elements = 0 

s3_pupilDiam_tr = np.nan_to_num(s3_pupilDiam_tr, nan=0)
s3_pupilDiam_test = np.nan_to_num(s3_pupilDiam_test, nan=0)
s4_pupilDiam_tr = np.nan_to_num(s4_pupilDiam_tr, nan=0)
s4_pupilDiam_test = np.nan_to_num(s4_pupilDiam_test, nan=0)
s5_pupilDiam_tr = np.nan_to_num(s5_pupilDiam_tr, nan=0)
s5_pupilDiam_test = np.nan_to_num(s5_pupilDiam_test, nan=0)

# s3_pupilDiam_tr = np.round(s3_pupilDiam_tr, 4)



# load the pupildiam data for stationary trials 

s3_pupilDiam_tr_st = np.load('./information/pupilDiam_st/A118_s3_pupilDiam_tr_st.npy')
s3_pupilDiam_test_st = np.load('./information/pupilDiam_st/A118_s3_pupilDiam_test_st.npy')
s4_pupilDiam_tr_st = np.load('./information/pupilDiam_st/A118_s4_pupilDiam_tr_st.npy')
s4_pupilDiam_test_st = np.load("./information/pupilDiam_st/A118_s4_pupilDiam_test_st.npy")
s5_pupilDiam_tr_st = np.load('./information/pupilDiam_st/A118_s5_pupilDiam_tr_st.npy')
s5_pupilDiam_test_st = np.load('./information/pupilDiam_st/A118_s5_pupilDiam_test_st.npy')

# nan = 0

s3_pupilDiam_tr_st = np.nan_to_num(s3_pupilDiam_tr_st, nan=0)
s3_pupilDiam_test_st = np.nan_to_num(s3_pupilDiam_test_st, nan=0)
s4_pupilDiam_tr_st = np.nan_to_num(s4_pupilDiam_tr_st, nan=0)
s4_pupilDiam_test_st = np.nan_to_num(s4_pupilDiam_test_st, nan=0)
s5_pupilDiam_tr_st = np.nan_to_num(s5_pupilDiam_tr_st, nan=0)
s5_pupilDiam_test_st = np.nan_to_num(s5_pupilDiam_test_st, nan=0)


# load stationary-filtered indices for training and testing

s3_indices_tr_st = np.load('./information/indices_stationary/A118_s3_test_st.npy', allow_pickle=True)
s3_indices_test_st = np.load('./information/indices_stationary/A118_s3_test_st.npy', allow_pickle=True)
s4_indices_tr_st = np.load('./information/indices_stationary/A118_s4_tr_st.npy', allow_pickle=True)
s4_indices_test_st = np.load('./information/indices_stationary/A118_s4_test_st.npy', allow_pickle=True)
s5_indices_tr_st = np.load('./information/indices_stationary/A118_s5_tr_st.npy', allow_pickle=True)
s5_indices_test_st = np.load('./information/indices_stationary/A118_s5_test_st.npy', allow_pickle=True)

# load neural data

s3_neuraldata_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s3_tr_st.npy')
s3_neuraldata_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s3_test_st.npy')
s4_neuraldata_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s4_tr_st.npy')
s4_neuraldata_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s4_test_st.npy')
s5_neuraldata_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s5_tr_st.npy')
s5_neuraldata_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s5_test_st.npy')

# load filmID data

s3_filmID_tr_st = np.load('./information/filmID_stationary/A118_s3_filmID_tr_st.npy')
s3_filmID_test_st = np.load('./information/filmID_stationary/A118_s3_filmID_test_st.npy')
s4_filmID_tr_st = np.load('./information/filmID_stationary/A118_s4_filmID_tr_st.npy')
s4_filmID_test_st = np.load('./information/filmID_stationary/A118_s4_filmID_test_st.npy')
s5_filmID_tr_st = np.load('./information/filmID_stationary/A118_s5_filmID_tr_st.npy')
s5_filmID_test_st = np.load('./information/filmID_stationary/A118_s5_filmID_test_st.npy')


# load the frameID data (has not been stationary filtered)

s3_frameID_tr = np.load('./information/frameID/A118_s3_frameID_tr.npy', allow_pickle=True)
s3_frameID_test = np.load('./information/frameID/A118_s3_frameID_test.npy', allow_pickle=True)

s4_frameID_tr = np.load('./information/frameID/A118_s4_frameID_tr.npy', allow_pickle=True)
s4_frameID_test = np.load('./information/frameID/A118_s4_frameID_test.npy', allow_pickle=True)

s5_frameID_tr = np.load('./information/frameID/A118_s5_frameID_tr.npy', allow_pickle=True)
s5_frameID_test = np.load('./information/frameID/A118_s5_frameID_test.npy', allow_pickle=True)

# load the frameID data (has been stationary filtered)

s3_frameID_tr_st = np.load('./information/frameID_stationary/A118_s3_frameID_tr_st.npy', allow_pickle=True)
s3_frameID_test_st = np.load('./information/frameID_stationary/A118_s3_frameID_test_st.npy', allow_pickle=True)

s4_frameID_tr_st = np.load('./information/frameID_stationary/A118_s4_frameID_tr_st.npy', allow_pickle=True)
s4_frameID_test_st = np.load('./information/frameID_stationary/A118_s4_frameID_test_st.npy', allow_pickle=True)

s5_frameID_tr_st = np.load('./information/frameID_stationary/A118_s5_frameID_tr_st.npy', allow_pickle=True)
s5_frameID_test_st = np.load('./information/frameID_stationary/A118_s5_frameID_test_st.npy', allow_pickle=True)


single_cebra_model = cebra.CEBRA(
    model_architecture = "offset10-model",
    batch_size = 516, 
    temperature_mode="auto",
    learning_rate = 0.001,
    max_iterations = 10000,
    max_adapt_iterations = 300,
    time_offsets = 10,
    output_dimension = 9, 
    device = device,
    verbose = True,
    conditional = 'time_delta',
)

# create deepcopy for initialising 12 models

models = []
for i in range(12):
    models.append(deepcopy(single_cebra_model))


# # train the models for non-stationary data for FilmID (./training_results/models_22)
# aux_f4.trainCebraBehaviour_dual(models[0], s3_neuraldata_tr, s3_pupilDiam_tr ,s3_filmID_tr, 3, 'm22_s3', s3_indices_tr, s3_indices_test, 'models_22')
# aux_f4.trainCebraBehaviour_dual(models[1], s4_neuraldata_tr, s4_pupilDiam_tr ,s4_filmID_tr, 4, 'm22_s4', s4_indices_tr, s4_indices_test, 'models_22')
# aux_f4.trainCebraBehaviour_dual(models[2], s5_neuraldata_tr, s5_pupilDiam_tr ,s5_filmID_tr, 5, 'm22_s5', s5_indices_tr, s5_indices_test, 'models_22')

# # train the models for stationary data for FilmID (./training_results/models_23)

# aux_f4.trainCebraBehaviour_dual(models[3], s3_neuraldata_tr_st, s3_pupilDiam_tr_st, s3_filmID_tr_st, 3, 'm23_s3', s3_indices_tr_st, s3_indices_test_st, 'models_23')
# aux_f4.trainCebraBehaviour_dual(models[4], s4_neuraldata_tr_st, s4_pupilDiam_tr_st, s4_filmID_tr_st, 4, 'm23_s4', s4_indices_tr_st, s4_indices_test_st, 'models_23')
# aux_f4.trainCebraBehaviour_dual(models[5], s5_neuraldata_tr_st, s5_pupilDiam_tr_st, s5_filmID_tr_st, 5, 'm23_s5', s5_indices_tr_st, s5_indices_test_st, 'models_23')    

# # train the models for non-stationary data for FrameID (./training_results/models_24) -- DONE

# aux_f4.trainCebraBehaviour_dual(models[6], s3_neuraldata_tr, s3_pupilDiam_tr,s3_frameID_tr, 3, 'm24_s3', s3_indices_tr, s3_indices_test, 'models_24')
# aux_f4.trainCebraBehaviour_dual(models[7], s4_neuraldata_tr, s4_pupilDiam_tr,s4_frameID_tr, 4, 'm24_s4', s4_indices_tr, s4_indices_test, 'models_24')
# aux_f4.trainCebraBehaviour_dual(models[8], s5_neuraldata_tr, s5_pupilDiam_tr,s5_frameID_tr, 5, 'm24_s5', s5_indices_tr, s5_indices_test, 'models_24')

# # train the models for stationary data for FrameID (./training_results/models_25) -- DONE

# aux_f4.trainCebraBehaviour_dual(models[9],  s3_neuraldata_tr_st, s3_pupilDiam_tr_st,s3_frameID_tr_st, 3, 'm25_s3', s3_indices_tr_st, s3_indices_test_st, 'models_25')
aux_f4.trainCebraBehaviour_dual(models[10], s4_neuraldata_tr_st, s4_pupilDiam_tr_st,s4_frameID_tr_st, 4, 'm25_s4', s4_indices_tr_st, s4_indices_test_st, 'models_25')
aux_f4.trainCebraBehaviour_dual(models[11], s5_neuraldata_tr_st, s5_pupilDiam_tr_st,s5_frameID_tr_st, 5, 'm25_s5', s5_indices_tr_st, s5_indices_test_st, 'models_25')

## --------     KNN using models trained with pupildiam data  -------------

s3_model_22 = CEBRA.load('./training_results/models_22/cebra_model_m22_s3.pt', map_location=torch.device("cpu"))
s4_model_22 = CEBRA.load('./training_results/models_22/cebra_model_m22_s4.pt', map_location=torch.device("cpu"))
s5_model_22 = CEBRA.load('./training_results/models_22/cebra_model_m22_s5.pt', map_location=torch.device("cpu"))

s3_model_23 = CEBRA.load('./training_results/models_23/cebra_model_m23_s3.pt', map_location=torch.device("cpu"))
s4_model_23 = CEBRA.load('./training_results/models_23/cebra_model_m23_s4.pt', map_location=torch.device("cpu"))
s5_model_23 = CEBRA.load('./training_results/models_23/cebra_model_m23_s5.pt', map_location=torch.device("cpu"))

s3_model_24 = CEBRA.load('./training_results/models_24/cebra_model_m24_s3.pt', map_location=torch.device("cpu"))
s4_model_24 = CEBRA.load('./training_results/models_24/cebra_model_m24_s4.pt', map_location=torch.device("cpu"))
s5_model_24 = CEBRA.load('./training_results/models_24/cebra_model_m24_s5.pt', map_location=torch.device("cpu"))

s3_model_25 = CEBRA.load('./training_results/models_25/cebra_model_m25_s3.pt', map_location=torch.device("cpu"))
s4_model_25 = CEBRA.load('./training_results/models_25/cebra_model_m25_s4.pt', map_location=torch.device("cpu"))
s5_model_25 = CEBRA.load('./training_results/models_25/cebra_model_m25_s5.pt', map_location=torch.device("cpu"))

# use cKNN() function to cross-check the cKNN for all models (pupilDiam)

# non-stationary filtered data
s3_3_acc_filmID_cknn_p = aux_f4.cKNN(s3_model_22, s3_neuraldata_tr, s3_neuraldata_test, s3_filmID_tr, s3_filmID_test)
s3_4_acc_filmID_cknn_p = aux_f4.cKNN(s3_model_22, s4_neuraldata_tr, s4_neuraldata_test, s4_filmID_tr, s4_filmID_test)
s3_5_acc_filmID_cknn_p = aux_f4.cKNN(s3_model_22, s5_neuraldata_tr, s5_neuraldata_test, s5_filmID_tr, s5_filmID_test)

s4_3_acc_filmID_cknn_p = aux_f4.cKNN(s4_model_22, s3_neuraldata_tr, s3_neuraldata_test, s3_filmID_tr, s3_filmID_test)
s4_4_acc_filmID_cknn_p = aux_f4.cKNN(s4_model_22, s4_neuraldata_tr, s4_neuraldata_test, s4_filmID_tr, s4_filmID_test)
s4_5_acc_filmID_cknn_p = aux_f4.cKNN(s4_model_22, s5_neuraldata_tr, s5_neuraldata_test, s5_filmID_tr, s5_filmID_test)

s5_3_acc_filmID_cknn_p = aux_f4.cKNN(s5_model_22, s3_neuraldata_tr, s3_neuraldata_test, s3_filmID_tr, s3_filmID_test)
s5_4_acc_filmID_cknn_p = aux_f4.cKNN(s5_model_22, s4_neuraldata_tr, s4_neuraldata_test, s4_filmID_tr, s4_filmID_test)
s5_5_acc_filmID_cknn_p = aux_f4.cKNN(s5_model_22, s5_neuraldata_tr, s5_neuraldata_test, s5_filmID_tr, s5_filmID_test)

accuracy_scores_filmID_cKNN_p = [s3_3_acc_filmID_cknn_p, s3_4_acc_filmID_cknn_p, s3_5_acc_filmID_cknn_p, s4_3_acc_filmID_cknn_p, s4_4_acc_filmID_cknn_p, s4_5_acc_filmID_cknn_p, s5_3_acc_filmID_cknn_p, s5_4_acc_filmID_cknn_p, s5_5_acc_filmID_cknn_p]

# stationary filtered data
s3_3_acc_filmID_cknn_st_p = aux_f4.cKNN(s3_model_23, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_filmID_tr_st, s3_filmID_test_st)
s3_4_acc_filmID_cknn_st_p = aux_f4.cKNN(s3_model_23, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_filmID_tr_st, s4_filmID_test_st)
s3_5_acc_filmID_cknn_st_p= aux_f4.cKNN(s3_model_23, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_filmID_tr_st, s5_filmID_test_st)

s4_3_acc_filmID_cknn_st_p = aux_f4.cKNN(s4_model_23, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_filmID_tr_st, s3_filmID_test_st)
s4_4_acc_filmID_cknn_st_p = aux_f4.cKNN(s4_model_23, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_filmID_tr_st, s4_filmID_test_st)
s4_5_acc_filmID_cknn_st_p = aux_f4.cKNN(s4_model_23, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_filmID_tr_st, s5_filmID_test_st)

s5_3_acc_filmID_cknn_st_p = aux_f4.cKNN(s5_model_23, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_filmID_tr_st, s3_filmID_test_st)
s5_4_acc_filmID_cknn_st_p = aux_f4.cKNN(s5_model_23, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_filmID_tr_st, s4_filmID_test_st)
s5_5_acc_filmID_cknn_st_p = aux_f4.cKNN(s5_model_23, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_filmID_tr_st, s5_filmID_test_st)

accuracy_scores_filmID_cKNN_st_p = [s3_3_acc_filmID_cknn_st_p, s3_4_acc_filmID_cknn_st_p, s3_5_acc_filmID_cknn_st_p, s4_3_acc_filmID_cknn_st_p, s4_4_acc_filmID_cknn_st_p, s4_5_acc_filmID_cknn_st_p, s5_3_acc_filmID_cknn_st_p, s5_4_acc_filmID_cknn_st_p, s5_5_acc_filmID_cknn_st_p]

# repeat for FrameID

# non-stationary filtered data
s3_3_acc_frameID_cknn_p = aux_f4.cKNN(s3_model_24, s3_neuraldata_tr, s3_neuraldata_test, s3_frameID_tr, s3_frameID_test)
s3_4_acc_frameID_cknn_p = aux_f4.cKNN(s3_model_24, s4_neuraldata_tr, s4_neuraldata_test, s4_frameID_tr, s4_frameID_test)
s3_5_acc_frameID_cknn_p = aux_f4.cKNN(s3_model_24, s5_neuraldata_tr, s5_neuraldata_test, s5_frameID_tr, s5_frameID_test)

s4_3_acc_frameID_cknn_p = aux_f4.cKNN(s4_model_24, s3_neuraldata_tr, s3_neuraldata_test, s3_frameID_tr, s3_frameID_test)
s4_4_acc_frameID_cknn_p = aux_f4.cKNN(s4_model_24, s4_neuraldata_tr, s4_neuraldata_test, s4_frameID_tr, s4_frameID_test)
s4_5_acc_frameID_cknn_p = aux_f4.cKNN(s4_model_24, s5_neuraldata_tr, s5_neuraldata_test, s5_frameID_tr, s5_frameID_test)

s5_3_acc_frameID_cknn_p = aux_f4.cKNN(s5_model_24, s3_neuraldata_tr, s3_neuraldata_test, s3_frameID_tr, s3_frameID_test)
s5_4_acc_frameID_cknn_p = aux_f4.cKNN(s5_model_24, s4_neuraldata_tr, s4_neuraldata_test, s4_frameID_tr, s4_frameID_test)
s5_5_acc_frameID_cknn_p = aux_f4.cKNN(s5_model_24, s5_neuraldata_tr, s5_neuraldata_test, s5_frameID_tr, s5_frameID_test)

accuracy_scores_frameID_cKNN_p = [s3_3_acc_frameID_cknn_p, s3_4_acc_frameID_cknn_p, s3_5_acc_frameID_cknn_p, s4_3_acc_frameID_cknn_p, s4_4_acc_frameID_cknn_p, s4_5_acc_frameID_cknn_p, s5_3_acc_frameID_cknn_p, s5_4_acc_frameID_cknn_p, s5_5_acc_frameID_cknn_p]


# stationary filtered data
s3_3_acc_frameID_cknn_st_p = aux_f4.cKNN(s3_model_25, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_frameID_tr_st, s3_frameID_test_st)
s3_4_acc_frameID_cknn_st_p = aux_f4.cKNN(s3_model_25, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_frameID_tr_st, s4_frameID_test_st)
s3_5_acc_frameID_cknn_st_p = aux_f4.cKNN(s3_model_25, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_frameID_tr_st, s5_frameID_test_st)

s4_3_acc_frameID_cknn_st_p = aux_f4.cKNN(s4_model_25, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_frameID_tr_st, s3_frameID_test_st)
s4_4_acc_frameID_cknn_st_p = aux_f4.cKNN(s4_model_25, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_frameID_tr_st, s4_frameID_test_st)
s4_5_acc_frameID_cknn_st_p = aux_f4.cKNN(s4_model_25, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_frameID_tr_st, s5_frameID_test_st)

s5_3_acc_frameID_cknn_st_p = aux_f4.cKNN(s5_model_25, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_frameID_tr_st, s3_frameID_test_st)
s5_4_acc_frameID_cknn_st_p = aux_f4.cKNN(s5_model_25, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_frameID_tr_st, s4_frameID_test_st)
s5_5_acc_frameID_cknn_st_p = aux_f4.cKNN(s5_model_25, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_frameID_tr_st, s5_frameID_test_st)

accuracy_scores_frameID_cKNN_st_p = [s3_3_acc_frameID_cknn_st_p, s3_4_acc_frameID_cknn_st_p, s3_5_acc_frameID_cknn_st_p, s4_3_acc_frameID_cknn_st_p, s4_4_acc_frameID_cknn_st_p, s4_5_acc_frameID_cknn_st_p, s5_3_acc_frameID_cknn_st_p, s5_4_acc_frameID_cknn_st_p, s5_5_acc_frameID_cknn_st_p]


# ------------- KNN USING MODELS 14 15 16 17 FOR COMPARISON (I.E NO PUPILDIAM DATA) ------------

# load the models for raw data (filmID)

s3_model_5 = CEBRA.load('./training_results/models_10/cebra_model_m10_s3.pt', map_location=torch.device("cpu"))
s4_model_5 = CEBRA.load('./training_results/models_10/cebra_model_m10_s4.pt', map_location=torch.device("cpu"))
s5_model_5 = CEBRA.load('./training_results/models_10/cebra_model_m10_s5.pt', map_location=torch.device("cpu"))

# load the models for stationary data (filmID)
s3_model_6 = CEBRA.load('./training_results/models_11/cebra_model_m11_s3.pt', map_location=torch.device("cpu"))
s4_model_6 = CEBRA.load('./training_results/models_11/cebra_model_m11_s4.pt', map_location=torch.device("cpu"))
s5_model_6 = CEBRA.load('./training_results/models_11/cebra_model_m11_s5.pt', map_location=torch.device("cpu"))

# use cKNN() function to cross-check the cKNN for all models (filmID)

# non-stationary filtered data
s3_3_acc_filmID_cknn = aux_f4.cKNN(s3_model_5, s3_neuraldata_tr, s3_neuraldata_test, s3_filmID_tr, s3_filmID_test)
s3_4_acc_filmID_cknn = aux_f4.cKNN(s3_model_5, s4_neuraldata_tr, s4_neuraldata_test, s4_filmID_tr, s4_filmID_test)
s3_5_acc_filmID_cknn = aux_f4.cKNN(s3_model_5, s5_neuraldata_tr, s5_neuraldata_test, s5_filmID_tr, s5_filmID_test)

s4_3_acc_filmID_cknn = aux_f4.cKNN(s4_model_5, s3_neuraldata_tr, s3_neuraldata_test, s3_filmID_tr, s3_filmID_test)
s4_4_acc_filmID_cknn = aux_f4.cKNN(s4_model_5, s4_neuraldata_tr, s4_neuraldata_test, s4_filmID_tr, s4_filmID_test)
s4_5_acc_filmID_cknn = aux_f4.cKNN(s4_model_5, s5_neuraldata_tr, s5_neuraldata_test, s5_filmID_tr, s5_filmID_test)

s5_3_acc_filmID_cknn = aux_f4.cKNN(s5_model_5, s3_neuraldata_tr, s3_neuraldata_test, s3_filmID_tr, s3_filmID_test)
s5_4_acc_filmID_cknn = aux_f4.cKNN(s5_model_5, s4_neuraldata_tr, s4_neuraldata_test, s4_filmID_tr, s4_filmID_test)
s5_5_acc_filmID_cknn = aux_f4.cKNN(s5_model_5, s5_neuraldata_tr, s5_neuraldata_test, s5_filmID_tr, s5_filmID_test)

accuracy_scores_filmID_cKNN = [s3_3_acc_filmID_cknn, s3_4_acc_filmID_cknn, s3_5_acc_filmID_cknn, s4_3_acc_filmID_cknn, s4_4_acc_filmID_cknn, s4_5_acc_filmID_cknn, s5_3_acc_filmID_cknn, s5_4_acc_filmID_cknn, s5_5_acc_filmID_cknn]


# stationary filtered data
s3_3_acc_filmID_cknn_st = aux_f4.cKNN(s3_model_6, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_filmID_tr_st, s3_filmID_test_st)
s3_4_acc_filmID_cknn_st = aux_f4.cKNN(s3_model_6, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_filmID_tr_st, s4_filmID_test_st)
s3_5_acc_filmID_cknn_st = aux_f4.cKNN(s3_model_6, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_filmID_tr_st, s5_filmID_test_st)

s4_3_acc_filmID_cknn_st = aux_f4.cKNN(s4_model_6, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_filmID_tr_st, s3_filmID_test_st)
s4_4_acc_filmID_cknn_st = aux_f4.cKNN(s4_model_6, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_filmID_tr_st, s4_filmID_test_st)
s4_5_acc_filmID_cknn_st = aux_f4.cKNN(s4_model_6, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_filmID_tr_st, s5_filmID_test_st)

s5_3_acc_filmID_cknn_st = aux_f4.cKNN(s5_model_6, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_filmID_tr_st, s3_filmID_test_st)
s5_4_acc_filmID_cknn_st = aux_f4.cKNN(s5_model_6, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_filmID_tr_st, s4_filmID_test_st)
s5_5_acc_filmID_cknn_st = aux_f4.cKNN(s5_model_6, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_filmID_tr_st, s5_filmID_test_st)

accuracy_scores_filmID_cKNN_st = [s3_3_acc_filmID_cknn_st, s3_4_acc_filmID_cknn_st, s3_5_acc_filmID_cknn_st, s4_3_acc_filmID_cknn_st, s4_4_acc_filmID_cknn_st, s4_5_acc_filmID_cknn_st, s5_3_acc_filmID_cknn_st, s5_4_acc_filmID_cknn_st, s5_5_acc_filmID_cknn_st]

# ------ repeat for frameID models -----

# load the models for raw data (frameID)

s3_model_7 = CEBRA.load('./training_results/models_12/cebra_model_m12_s3.pt', map_location=torch.device("cpu"))
s4_model_7 = CEBRA.load('./training_results/models_12/cebra_model_m12_s4.pt', map_location=torch.device("cpu"))
s5_model_7 = CEBRA.load('./training_results/models_12/cebra_model_m12_s5.pt', map_location=torch.device("cpu"))

# load the models for stationary data (frameID)

s3_model_8 = CEBRA.load('./training_results/models_13/cebra_model_m13_s3.pt', map_location=torch.device("cpu"))
s4_model_8 = CEBRA.load('./training_results/models_13/cebra_model_m13_s4.pt', map_location=torch.device("cpu"))
s5_model_8 = CEBRA.load('./training_results/models_13/cebra_model_m13_s5.pt', map_location=torch.device("cpu"))

# use cKNN() function to cross-check the cKNN for all models (frameID)

# non-stationary filtered data
s3_3_acc_frameID_cknn = aux_f4.cKNN(s3_model_7, s3_neuraldata_tr, s3_neuraldata_test, s3_frameID_tr, s3_frameID_test)
s3_4_acc_frameID_cknn = aux_f4.cKNN(s3_model_7, s4_neuraldata_tr, s4_neuraldata_test, s4_frameID_tr, s4_frameID_test)
s3_5_acc_frameID_cknn = aux_f4.cKNN(s3_model_7, s5_neuraldata_tr, s5_neuraldata_test, s5_frameID_tr, s5_frameID_test)

s4_3_acc_frameID_cknn = aux_f4.cKNN(s4_model_7, s3_neuraldata_tr, s3_neuraldata_test, s3_frameID_tr, s3_frameID_test)
s4_4_acc_frameID_cknn = aux_f4.cKNN(s4_model_7, s4_neuraldata_tr, s4_neuraldata_test, s4_frameID_tr, s4_frameID_test)
s4_5_acc_frameID_cknn = aux_f4.cKNN(s4_model_7, s5_neuraldata_tr, s5_neuraldata_test, s5_frameID_tr, s5_frameID_test)

s5_3_acc_frameID_cknn = aux_f4.cKNN(s5_model_7, s3_neuraldata_tr, s3_neuraldata_test, s3_frameID_tr, s3_frameID_test)
s5_4_acc_frameID_cknn = aux_f4.cKNN(s5_model_7, s4_neuraldata_tr, s4_neuraldata_test, s4_frameID_tr, s4_frameID_test)
s5_5_acc_frameID_cknn = aux_f4.cKNN(s5_model_7, s5_neuraldata_tr, s5_neuraldata_test, s5_frameID_tr, s5_frameID_test)

accuracy_scores_frameID_cKNN = [s3_3_acc_frameID_cknn, s3_4_acc_frameID_cknn, s3_5_acc_frameID_cknn, s4_3_acc_frameID_cknn, s4_4_acc_frameID_cknn, s4_5_acc_frameID_cknn, s5_3_acc_frameID_cknn, s5_4_acc_frameID_cknn, s5_5_acc_frameID_cknn]


# stationary filtered data
s3_3_acc_frameID_cknn_st = aux_f4.cKNN(s3_model_8, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_frameID_tr_st, s3_frameID_test_st)
s3_4_acc_frameID_cknn_st = aux_f4.cKNN(s3_model_8, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_frameID_tr_st, s4_frameID_test_st)
s3_5_acc_frameID_cknn_st = aux_f4.cKNN(s3_model_8, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_frameID_tr_st, s5_frameID_test_st)

s4_3_acc_frameID_cknn_st = aux_f4.cKNN(s4_model_8, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_frameID_tr_st, s3_frameID_test_st)
s4_4_acc_frameID_cknn_st = aux_f4.cKNN(s4_model_8, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_frameID_tr_st, s4_frameID_test_st)
s4_5_acc_frameID_cknn_st = aux_f4.cKNN(s4_model_8, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_frameID_tr_st, s5_frameID_test_st)

s5_3_acc_frameID_cknn_st = aux_f4.cKNN(s5_model_8, s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_frameID_tr_st, s3_frameID_test_st)
s5_4_acc_frameID_cknn_st = aux_f4.cKNN(s5_model_8, s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_frameID_tr_st, s4_frameID_test_st)
s5_5_acc_frameID_cknn_st = aux_f4.cKNN(s5_model_8, s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_frameID_tr_st, s5_frameID_test_st)

accuracy_scores_frameID_cKNN_st = [s3_3_acc_frameID_cknn_st, s3_4_acc_frameID_cknn_st, s3_5_acc_frameID_cknn_st, s4_3_acc_frameID_cknn_st, s4_4_acc_frameID_cknn_st, s4_5_acc_frameID_cknn_st, s5_3_acc_frameID_cknn_st, s5_4_acc_frameID_cknn_st, s5_5_acc_frameID_cknn_st]

## ----- PLOT COMPARISON ------- #

# 1. KNN v cKNN for filmID data - not stationary filtered 
aux_f4.plotKNN_pupilDiam(accuracy_scores_filmID_cKNN, accuracy_scores_filmID_cKNN_p, '- filmID - not stationary filtered')

# 2. KNN v cKNN for filmID data - stationary filtered
aux_f4.plotKNN_pupilDiam(accuracy_scores_filmID_cKNN_st, accuracy_scores_filmID_cKNN_st_p , '- filmID - stationary filtered')

# 3. KNN v cKNN for frameID data - not stationary filtered
aux_f4.plotKNN_pupilDiam(accuracy_scores_frameID_cKNN, accuracy_scores_frameID_cKNN_p, '- frameID - not stationary filtered')

# 4. KNN v cKNN for frameID data - stationary filtered
aux_f4.plotKNN_pupilDiam(accuracy_scores_frameID_cKNN_st, accuracy_scores_frameID_cKNN_st_p, '- frameID - stationary filtered')

plt.show()