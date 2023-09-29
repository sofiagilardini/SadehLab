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

# load all data 

""" 1. Neural data not shuffled
    2. FilmID not shuffled 
    3. FrameID not shuffled 

    4. Neural data not shuffled stationary filtered
    5. FilmID not shuffled stationary filtered
    6. FrameID not shuffled stationary filtered

    7. Neural data shuffled
    8. FilmID shuffled
    9. FrameID shuffled

    10. Neural data shuffled stationary filtered
    11. FilmID shuffled stationary filtered
    12. FrameID shuffled stationary filtered

"""

# # 1. Neural data not shuffled

# s3_neuraldata_tr = np.load('./information/neuraldata/A118_neuraldata_session3_tr.npy', allow_pickle=True)
# s3_neuraldata_test = np.load('./information/neuraldata/A118_neuraldata_session3_test.npy', allow_pickle=True)
# s4_neuraldata_tr = np.load('./information/neuraldata/A118_neuraldata_session4_tr.npy', allow_pickle=True)
# s4_neuraldata_test = np.load('./information/neuraldata/A118_neuraldata_session4_test.npy', allow_pickle=True)
# s5_neuraldata_tr = np.load('./information/neuraldata/A118_neuraldata_session5_tr.npy', allow_pickle=True)
# s5_neuraldata_test = np.load('./information/neuraldata/A118_neuraldata_session5_test.npy', allow_pickle=True)

# # 2. FilmID not shuffled

# s3_filmID_tr = np.load('./information/filmID/A118_s3_filmID_tr.npy', allow_pickle=True)
# s3_filmID_test = np.load('./information/filmID/A118_s3_filmID_test.npy', allow_pickle=True)
# s4_filmID_tr = np.load('./information/filmID/A118_s4_filmID_tr.npy', allow_pickle=True)
# s4_filmID_test = np.load('./information/filmID/A118_s4_filmID_test.npy', allow_pickle=True)
# s5_filmID_tr = np.load('./information/filmID/A118_s5_filmID_tr.npy', allow_pickle=True)
# s5_filmID_test = np.load('./information/filmID/A118_s5_filmID_test.npy', allow_pickle=True)

# # 3. FrameID not shuffled

# s3_frameID_tr = np.load('./information/frameID/A118_s3_frameID_tr.npy', allow_pickle=True)
# s3_frameID_test = np.load('./information/frameID/A118_s3_frameID_test.npy', allow_pickle=True)
# s4_frameID_tr = np.load('./information/frameID/A118_s4_frameID_tr.npy', allow_pickle=True)
# s4_frameID_test = np.load('./information/frameID/A118_s4_frameID_test.npy', allow_pickle=True)
# s5_frameID_tr = np.load('./information/frameID/A118_s5_frameID_tr.npy', allow_pickle=True)
# s5_frameID_test = np.load('./information/frameID/A118_s5_frameID_test.npy', allow_pickle=True)

# # 4. Neural data not shuffled stationary filtered

# s3_neuraldata_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s3_tr_st.npy')
# s3_neuraldata_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s3_test_st.npy')
# s4_neuraldata_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s4_tr_st.npy')
# s4_neuraldata_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s4_test_st.npy')
# s5_neuraldata_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s5_tr_st.npy')
# s5_neuraldata_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s5_test_st.npy')

# # 5. FilmID not shuffled stationary filtered

# s3_filmID_tr_st = np.load('./information/filmID_stationary/A118_s3_filmID_tr_st.npy')
# s3_filmID_test_st = np.load('./information/filmID_stationary/A118_s3_filmID_test_st.npy')
# s4_filmID_tr_st = np.load('./information/filmID_stationary/A118_s4_filmID_tr_st.npy')
# s4_filmID_test_st = np.load('./information/filmID_stationary/A118_s4_filmID_test_st.npy')
# s5_filmID_tr_st = np.load('./information/filmID_stationary/A118_s5_filmID_tr_st.npy')
# s5_filmID_test_st = np.load('./information/filmID_stationary/A118_s5_filmID_test_st.npy')

# # 6. FrameID not shuffled stationary filtered

# s3_frameID_tr_st = np.load('./information/frameID_stationary/A118_s3_frameID_tr_st.npy', allow_pickle=True)
# s3_frameID_test_st = np.load('./information/frameID_stationary/A118_s3_frameID_test_st.npy', allow_pickle=True)
# s4_frameID_tr_st = np.load('./information/frameID_stationary/A118_s4_frameID_tr_st.npy', allow_pickle=True)
# s4_frameID_test_st = np.load('./information/frameID_stationary/A118_s4_frameID_test_st.npy', allow_pickle=True)
# s5_frameID_tr_st = np.load('./information/frameID_stationary/A118_s5_frameID_tr_st.npy', allow_pickle=True)
# s5_frameID_test_st = np.load('./information/frameID_stationary/A118_s5_frameID_test_st.npy', allow_pickle=True)

# 7. Neural data shuffled

s3_neuraldata_tr_sh = np.load('./information/neuraldata_shuffled/A118_s3_neuraldata_tr_sh.npy', allow_pickle=True)
s3_neuraldata_test_sh = np.load('./information/neuraldata_shuffled/A118_s3_neuraldata_test_sh.npy', allow_pickle=True)
s4_neuraldata_tr_sh = np.load('./information/neuraldata_shuffled/A118_s4_neuraldata_tr_sh.npy', allow_pickle=True)
s4_neuraldata_test_sh = np.load('./information/neuraldata_shuffled/A118_s4_neuraldata_test_sh.npy', allow_pickle=True)
s5_neuraldata_tr_sh = np.load('./information/neuraldata_shuffled/A118_s5_neuraldata_tr_sh.npy', allow_pickle=True)
s5_neuraldata_test_sh = np.load('./information/neuraldata_shuffled/A118_s5_neuraldata_test_sh.npy', allow_pickle=True)

# 8. FilmID shuffled

s3_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s3_filmID_tr_sh.npy', allow_pickle=True)
s3_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s3_filmID_test_sh.npy', allow_pickle=True)
s4_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s4_filmID_tr_sh.npy', allow_pickle=True)
s4_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s4_filmID_test_sh.npy', allow_pickle=True)
s5_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s5_filmID_tr_sh.npy', allow_pickle=True)
s5_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s5_filmID_test_sh.npy', allow_pickle=True)

# 9. FrameID shuffled

s3_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s3_frameID_tr_sh.npy', allow_pickle=True)
s3_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s3_frameID_test_sh.npy', allow_pickle=True)
s4_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s4_frameID_tr_sh.npy', allow_pickle=True)
s4_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s4_frameID_test_sh.npy', allow_pickle=True)
s5_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s5_frameID_tr_sh.npy', allow_pickle=True)
s5_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s5_frameID_test_sh.npy', allow_pickle=True)

# 10. Neural data shuffled stationary filtered

s3_neuraldata_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s3_neuraldata_tr_st_sh.npy', allow_pickle=True)
s3_neuraldata_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s3_neuraldata_test_st_sh.npy', allow_pickle=True)
s4_neuraldata_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s4_neuraldata_tr_st_sh.npy', allow_pickle=True)
s4_neuraldata_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s4_neuraldata_test_st_sh.npy', allow_pickle=True)
s5_neuraldata_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s5_neuraldata_tr_st_sh.npy', allow_pickle=True)
s5_neuraldata_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s5_neuraldata_test_st_sh.npy', allow_pickle=True)

# 11. FilmID shuffled stationary filtered

s3_filmID_tr_sh_st = np.load('./information/filmID_st_shuffled/A118_s3_filmID_tr_st_sh.npy', allow_pickle=True)
s3_filmID_test_sh_st = np.load('./information/filmID_st_shuffled/A118_s3_filmID_test_st_sh.npy', allow_pickle=True)
s4_filmID_tr_sh_st = np.load('./information/filmID_st_shuffled/A118_s4_filmID_tr_st_sh.npy', allow_pickle=True)
s4_filmID_test_sh_st = np.load('./information/filmID_st_shuffled/A118_s4_filmID_test_st_sh.npy', allow_pickle=True)
s5_filmID_tr_sh_st = np.load('./information/filmID_st_shuffled/A118_s5_filmID_tr_st_sh.npy', allow_pickle=True)
s5_filmID_test_sh_st = np.load('./information/filmID_st_shuffled/A118_s5_filmID_test_st_sh.npy', allow_pickle=True)

# 12. FrameID shuffled stationary filtered

s3_frameID_tr_sh_st = np.load('./information/frameID_st_shuffled/A118_s3_frameID_tr_st_sh.npy', allow_pickle=True)
s3_frameID_test_sh_st = np.load('./information/frameID_st_shuffled/A118_s3_frameID_test_st_sh.npy', allow_pickle=True)
s4_frameID_tr_sh_st = np.load('./information/frameID_st_shuffled/A118_s4_frameID_tr_st_sh.npy', allow_pickle=True)
s4_frameID_test_sh_st = np.load('./information/frameID_st_shuffled/A118_s4_frameID_test_st_sh.npy', allow_pickle=True)
s5_frameID_tr_sh_st = np.load('./information/frameID_st_shuffled/A118_s5_frameID_tr_st_sh.npy', allow_pickle=True)
s5_frameID_test_sh_st = np.load('./information/frameID_st_shuffled/A118_s5_frameID_test_st_sh.npy', allow_pickle=True)

# do KNN on all data

# KNN for filmID using shuffled data

s3_3_acc_film = aux_f4.getAccuracy(s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_filmID_tr_sh, s3_filmID_test_sh, 's3_3_filmID_sh')
s3_4_acc_film = aux_f4.getAccuracy(s3_neuraldata_tr_sh, s4_neuraldata_test_sh, s3_filmID_tr_sh, s4_filmID_test_sh, 's3_4_filmID_sh')
s3_5_acc_film = aux_f4.getAccuracy(s3_neuraldata_tr_sh, s5_neuraldata_test_sh, s3_filmID_tr_sh, s5_filmID_test_sh, 's3_5_filmID_sh')

s4_3_acc_film = aux_f4.getAccuracy(s4_neuraldata_tr_sh, s3_neuraldata_test_sh, s4_filmID_tr_sh, s3_filmID_test_sh, 's4_3_filmID_sh')
s4_4_acc_film = aux_f4.getAccuracy(s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_filmID_tr_sh, s4_filmID_test_sh, 's4_4_filmID_sh')
s4_5_acc_film = aux_f4.getAccuracy(s4_neuraldata_tr_sh, s5_neuraldata_test_sh, s4_filmID_tr_sh, s5_filmID_test_sh, 's4_5_filmID_sh')

s5_3_acc_film = aux_f4.getAccuracy(s5_neuraldata_tr_sh, s3_neuraldata_test_sh, s5_filmID_tr_sh, s3_filmID_test_sh, 's5_3_filmID_sh')
s5_4_acc_film = aux_f4.getAccuracy(s5_neuraldata_tr_sh, s4_neuraldata_test_sh, s5_filmID_tr_sh, s4_filmID_test_sh, 's5_4_filmID_sh')
s5_5_acc_film = aux_f4.getAccuracy(s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_filmID_tr_sh, s5_filmID_test_sh, 's5_5_filmID_sh')

accuracy_scores_filmID_KNN = [s3_3_acc_film, s3_4_acc_film, s3_5_acc_film, s4_3_acc_film, s4_4_acc_film, s4_5_acc_film, s5_3_acc_film, s5_4_acc_film, s5_5_acc_film]

# KNN for filmID using shuffled data stationary filtered

s3_3_acc_film_st = aux_f4.getAccuracy(s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_filmID_tr_sh_st, s3_filmID_test_sh_st, 's3_3_filmID_sh_st')
s3_4_acc_film_st = aux_f4.getAccuracy(s3_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s3_filmID_tr_sh_st, s4_filmID_test_sh_st, 's3_4_filmID_sh_st')
s3_5_acc_film_st = aux_f4.getAccuracy(s3_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s3_filmID_tr_sh_st, s5_filmID_test_sh_st, 's3_5_filmID_sh_st')

s4_3_acc_film_st = aux_f4.getAccuracy(s4_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s4_filmID_tr_sh_st, s3_filmID_test_sh_st, 's4_3_filmID_sh_st')
s4_4_acc_film_st = aux_f4.getAccuracy(s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_filmID_tr_sh_st, s4_filmID_test_sh_st, 's4_4_filmID_sh_st')
s4_5_acc_film_st = aux_f4.getAccuracy(s4_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s4_filmID_tr_sh_st, s5_filmID_test_sh_st, 's4_5_filmID_sh_st')

s5_3_acc_film_st = aux_f4.getAccuracy(s5_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s5_filmID_tr_sh_st, s3_filmID_test_sh_st, 's5_3_filmID_sh_st')
s5_4_acc_film_st = aux_f4.getAccuracy(s5_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s5_filmID_tr_sh_st, s4_filmID_test_sh_st, 's5_4_filmID_sh_st')
s5_5_acc_film_st = aux_f4.getAccuracy(s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_filmID_tr_sh_st, s5_filmID_test_sh_st, 's5_5_filmID_sh_st')

accuracy_scores_filmID_st_KNN = [s3_3_acc_film_st, s3_4_acc_film_st, s3_5_acc_film_st, s4_3_acc_film_st, s4_4_acc_film_st, s4_5_acc_film_st, s5_3_acc_film_st, s5_4_acc_film_st, s5_5_acc_film_st]

# KNN for frameID using shuffled data

s3_3_acc_frame = aux_f4.getAccuracy(s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_frameID_tr_sh, s3_frameID_test_sh, 's3_3_frameID_sh')
s3_4_acc_frame = aux_f4.getAccuracy(s3_neuraldata_tr_sh, s4_neuraldata_test_sh, s3_frameID_tr_sh, s4_frameID_test_sh, 's3_4_frameID_sh')
s3_5_acc_frame = aux_f4.getAccuracy(s3_neuraldata_tr_sh, s5_neuraldata_test_sh, s3_frameID_tr_sh, s5_frameID_test_sh, 's3_5_frameID_sh')

s4_3_acc_frame = aux_f4.getAccuracy(s4_neuraldata_tr_sh, s3_neuraldata_test_sh, s4_frameID_tr_sh, s3_frameID_test_sh, 's4_3_frameID_sh')
s4_4_acc_frame = aux_f4.getAccuracy(s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_frameID_tr_sh, s4_frameID_test_sh, 's4_4_frameID_sh')
s4_5_acc_frame = aux_f4.getAccuracy(s4_neuraldata_tr_sh, s5_neuraldata_test_sh, s4_frameID_tr_sh, s5_frameID_test_sh, 's4_5_frameID_sh')

s5_3_acc_frame = aux_f4.getAccuracy(s5_neuraldata_tr_sh, s3_neuraldata_test_sh, s5_frameID_tr_sh, s3_frameID_test_sh, 's5_3_frameID_sh')
s5_4_acc_frame = aux_f4.getAccuracy(s5_neuraldata_tr_sh, s4_neuraldata_test_sh, s5_frameID_tr_sh, s4_frameID_test_sh, 's5_4_frameID_sh')
s5_5_acc_frame = aux_f4.getAccuracy(s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_frameID_tr_sh, s5_frameID_test_sh, 's5_5_frameID_sh')

accuracy_scores_frameID_KNN = [s3_3_acc_frame, s3_4_acc_frame, s3_5_acc_frame, s4_3_acc_frame, s4_4_acc_frame, s4_5_acc_frame, s5_3_acc_frame, s5_4_acc_frame, s5_5_acc_frame]
 
# KNN for frameID using shuffled data stationary filtered

s3_3_acc_frame_st = aux_f4.getAccuracy(s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_frameID_tr_sh_st, s3_frameID_test_sh_st, 's3_3_frameID_sh_st')
s3_4_acc_frame_st = aux_f4.getAccuracy(s3_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s3_frameID_tr_sh_st, s4_frameID_test_sh_st, 's3_4_frameID_sh_st')
s3_5_acc_frame_st = aux_f4.getAccuracy(s3_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s3_frameID_tr_sh_st, s5_frameID_test_sh_st, 's3_5_frameID_sh_st')

s4_3_acc_frame_st = aux_f4.getAccuracy(s4_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s4_frameID_tr_sh_st, s3_frameID_test_sh_st, 's4_3_frameID_sh_st')
s4_4_acc_frame_st = aux_f4.getAccuracy(s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_frameID_tr_sh_st, s4_frameID_test_sh_st, 's4_4_frameID_sh_st')
s4_5_acc_frame_st = aux_f4.getAccuracy(s4_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s4_frameID_tr_sh_st, s5_frameID_test_sh_st, 's4_5_frameID_sh_st')

s5_3_acc_frame_st = aux_f4.getAccuracy(s5_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s5_frameID_tr_sh_st, s3_frameID_test_sh_st, 's5_3_frameID_sh_st')
s5_4_acc_frame_st = aux_f4.getAccuracy(s5_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s5_frameID_tr_sh_st, s4_frameID_test_sh_st, 's5_4_frameID_sh_st')
s5_5_acc_frame_st = aux_f4.getAccuracy(s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_frameID_tr_sh_st, s5_frameID_test_sh_st, 's5_5_frameID_sh_st')

accuracy_scores_frameID_st_KNN = [s3_3_acc_frame_st, s3_4_acc_frame_st, s3_5_acc_frame_st, s4_3_acc_frame_st, s4_4_acc_frame_st, s4_5_acc_frame_st, s5_3_acc_frame_st, s5_4_acc_frame_st, s5_5_acc_frame_st]


# ---------- CEBRA MODEL DEFINITION AND TRAINING ------------ #

single_cebra_model = cebra.CEBRA(
    model_architecture = "offset10-model",
    batch_size = 512,
    temperature_mode="auto",
    learning_rate = 0.001,
    max_iterations = 10000,
    max_adapt_iterations = 300,
    time_offsets = 1,
    output_dimension = 9,
    device = device,
    verbose = True,
    conditional = 'time_delta'

    # models 18-22 are with conditional = time_delta
    # models 25- are with conditional = delta
)

models = []
for i in range(12):
    models.append(deepcopy(single_cebra_model))

# # -- FILMID TRAINING --- ##

# aux_f4.trainCebraBehaviour_sh(models[0], s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_filmID_tr_sh, s3_filmID_test_sh, 3, 'm26_s3', 'models_26')
# aux_f4.trainCebraBehaviour_sh(models[1], s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_filmID_tr_sh, s4_filmID_test_sh, 4, 'm26_s4', 'models_26')
# aux_f4.trainCebraBehaviour_sh(models[2], s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_filmID_tr_sh, s5_filmID_test_sh, 5, 'm26_s5',  'models_26')

# # train the models for stationary data for FilmID (./training_results/models_27)

# aux_f4.trainCebraBehaviour_sh(models[3], s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_filmID_tr_sh_st, s3_filmID_test_sh_st, 3, 'm27_s3', 'models_27')
# aux_f4.trainCebraBehaviour_sh(models[4], s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_filmID_tr_sh_st, s4_filmID_test_sh_st, 4, 'm27_s4', 'models_27')
# aux_f4.trainCebraBehaviour_sh(models[5], s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_filmID_tr_sh_st, s5_filmID_test_sh_st, 5, 'm27_s5', 'models_27')

# # #----- end of FilmID ------ #

# # -- FRAMEID TRAINING --- ##

# aux_f4.trainCebraBehaviour_sh(models[6], s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_frameID_tr_sh, s3_frameID_test_sh, 3, 'm28_s3', 'models_28')
# aux_f4.trainCebraBehaviour_sh(models[7], s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_frameID_tr_sh, s4_frameID_test_sh, 4, 'm28_s4', 'models_28')
# aux_f4.trainCebraBehaviour_sh(models[8], s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_frameID_tr_sh, s5_frameID_test_sh, 5, 'm28_s5', 'models_28')

# # train the models for stationary data for FrameID (./training_results/models_29)

# aux_f4.trainCebraBehaviour_sh(models[9], s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_frameID_tr_sh_st, s3_frameID_test_sh_st, 3, 'm29_s3', 'models_29')
# aux_f4.trainCebraBehaviour_sh(models[10], s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_frameID_tr_sh_st, s4_frameID_test_sh_st, 4, 'm29_s4', 'models_29')
# aux_f4.trainCebraBehaviour_sh(models[11], s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_frameID_tr_sh_st, s5_frameID_test_sh_st, 5, 'm29_s5', 'models_29')

# ----- end of FrameID ----- #

# ---- END OF CEBRA TRAINING ---- #

# ---- LOAD MODELS ----- #

# load the models for raw data (filmID)

s3_model_1 = CEBRA.load('./training_results/models_26/cebra_model_m26_s3.pt', map_location=torch.device("cpu"))
s4_model_1 = CEBRA.load('./training_results/models_26/cebra_model_m26_s4.pt', map_location=torch.device("cpu"))
s5_model_1 = CEBRA.load('./training_results/models_26/cebra_model_m26_s5.pt', map_location=torch.device("cpu"))

# load the models for stationary data (filmID)
s3_model_2 = CEBRA.load('./training_results/models_27/cebra_model_m27_s3.pt', map_location=torch.device("cpu"))
s4_model_2 = CEBRA.load('./training_results/models_27/cebra_model_m27_s4.pt', map_location=torch.device("cpu"))
s5_model_2 = CEBRA.load('./training_results/models_27/cebra_model_m27_s5.pt', map_location=torch.device("cpu"))

# load the models for raw data (frameID)

s3_model_3 = CEBRA.load('./training_results/models_28/cebra_model_m28_s3.pt', map_location=torch.device("cpu"))
s4_model_3 = CEBRA.load('./training_results/models_28/cebra_model_m28_s4.pt', map_location=torch.device("cpu"))
s5_model_3 = CEBRA.load('./training_results/models_28/cebra_model_m28_s5.pt', map_location=torch.device("cpu"))

# load the models for stationary data (frameID)

s3_model_4 = CEBRA.load('./training_results/models_29/cebra_model_m29_s3.pt', map_location=torch.device("cpu"))
s4_model_4 = CEBRA.load('./training_results/models_29/cebra_model_m29_s4.pt', map_location=torch.device("cpu"))
s5_model_4 = CEBRA.load('./training_results/models_29/cebra_model_m29_s5.pt', map_location=torch.device("cpu"))

# ---- END OF LOAD MODELS ----- #

# ---- CKNN ------ #

# use cKNN() function to cross-check the cKNN for all models (filmID)

# non-stationary filtered data

s3_3_acc_filmID_cknn = aux_f4.cKNN(s3_model_1, s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_filmID_tr_sh, s3_filmID_test_sh)
s3_4_acc_filmID_cknn = aux_f4.cKNN(s3_model_1, s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_filmID_tr_sh, s4_filmID_test_sh)
s3_5_acc_filmID_cknn = aux_f4.cKNN(s3_model_1, s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_filmID_tr_sh, s5_filmID_test_sh)

s4_3_acc_filmID_cknn = aux_f4.cKNN(s4_model_1, s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_filmID_tr_sh, s3_filmID_test_sh)
s4_4_acc_filmID_cknn = aux_f4.cKNN(s4_model_1, s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_filmID_tr_sh, s4_filmID_test_sh)
s4_5_acc_filmID_cknn = aux_f4.cKNN(s4_model_1, s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_filmID_tr_sh, s5_filmID_test_sh)

s5_3_acc_filmID_cknn = aux_f4.cKNN(s5_model_1, s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_filmID_tr_sh, s3_filmID_test_sh)
s5_4_acc_filmID_cknn = aux_f4.cKNN(s5_model_1, s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_filmID_tr_sh, s4_filmID_test_sh)
s5_5_acc_filmID_cknn = aux_f4.cKNN(s5_model_1, s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_filmID_tr_sh, s5_filmID_test_sh)

accuracy_scores_filmID_cKNN = [s3_3_acc_filmID_cknn, s3_4_acc_filmID_cknn, s3_5_acc_filmID_cknn, s4_3_acc_filmID_cknn, s4_4_acc_filmID_cknn, s4_5_acc_filmID_cknn, s5_3_acc_filmID_cknn, s5_4_acc_filmID_cknn, s5_5_acc_filmID_cknn]

# stationary filtered data

s3_3_acc_filmID_cknn_st = aux_f4.cKNN(s3_model_2, s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_filmID_tr_sh_st, s3_filmID_test_sh_st)
s3_4_acc_filmID_cknn_st = aux_f4.cKNN(s3_model_2, s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_filmID_tr_sh_st, s4_filmID_test_sh_st)
s3_5_acc_filmID_cknn_st = aux_f4.cKNN(s3_model_2, s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_filmID_tr_sh_st, s5_filmID_test_sh_st)

s4_3_acc_filmID_cknn_st = aux_f4.cKNN(s4_model_2, s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_filmID_tr_sh_st, s3_filmID_test_sh_st)
s4_4_acc_filmID_cknn_st = aux_f4.cKNN(s4_model_2, s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_filmID_tr_sh_st, s4_filmID_test_sh_st)
s4_5_acc_filmID_cknn_st = aux_f4.cKNN(s4_model_2, s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_filmID_tr_sh_st, s5_filmID_test_sh_st)

s5_3_acc_filmID_cknn_st = aux_f4.cKNN(s5_model_2, s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_filmID_tr_sh_st, s3_filmID_test_sh_st)
s5_4_acc_filmID_cknn_st = aux_f4.cKNN(s5_model_2, s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_filmID_tr_sh_st, s4_filmID_test_sh_st)
s5_5_acc_filmID_cknn_st = aux_f4.cKNN(s5_model_2, s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_filmID_tr_sh_st, s5_filmID_test_sh_st)

accuracy_scores_filmID_cKNN_st = [s3_3_acc_filmID_cknn_st, s3_4_acc_filmID_cknn_st, s3_5_acc_filmID_cknn_st, s4_3_acc_filmID_cknn_st, s4_4_acc_filmID_cknn_st, s4_5_acc_filmID_cknn_st, s5_3_acc_filmID_cknn_st, s5_4_acc_filmID_cknn_st, s5_5_acc_filmID_cknn_st]

# use cKNN() function to cross-check the cKNN for all models (frameID)

# non-stationary filtered data
s3_3_acc_frameID_cknn = aux_f4.cKNN(s3_model_3, s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_frameID_tr_sh, s3_frameID_test_sh)
s3_4_acc_frameID_cknn = aux_f4.cKNN(s3_model_3, s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_frameID_tr_sh, s4_frameID_test_sh)
s3_5_acc_frameID_cknn = aux_f4.cKNN(s3_model_3, s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_frameID_tr_sh, s5_frameID_test_sh)

s4_3_acc_frameID_cknn = aux_f4.cKNN(s4_model_3, s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_frameID_tr_sh, s3_frameID_test_sh)
s4_4_acc_frameID_cknn = aux_f4.cKNN(s4_model_3, s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_frameID_tr_sh, s4_frameID_test_sh)
s4_5_acc_frameID_cknn = aux_f4.cKNN(s4_model_3, s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_frameID_tr_sh, s5_frameID_test_sh)

s5_3_acc_frameID_cknn = aux_f4.cKNN(s5_model_3, s3_neuraldata_tr_sh, s3_neuraldata_test_sh, s3_frameID_tr_sh, s3_frameID_test_sh)
s5_4_acc_frameID_cknn = aux_f4.cKNN(s5_model_3, s4_neuraldata_tr_sh, s4_neuraldata_test_sh, s4_frameID_tr_sh, s4_frameID_test_sh)
s5_5_acc_frameID_cknn = aux_f4.cKNN(s5_model_3, s5_neuraldata_tr_sh, s5_neuraldata_test_sh, s5_frameID_tr_sh, s5_frameID_test_sh)

accuracy_scores_frameID_cKNN = [s3_3_acc_frameID_cknn, s3_4_acc_frameID_cknn, s3_5_acc_frameID_cknn, s4_3_acc_frameID_cknn, s4_4_acc_frameID_cknn, s4_5_acc_frameID_cknn, s5_3_acc_frameID_cknn, s5_4_acc_frameID_cknn, s5_5_acc_frameID_cknn]

# stationary filtered data
s3_3_acc_frameID_cknn_st = aux_f4.cKNN(s3_model_4, s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_frameID_tr_sh_st, s3_frameID_test_sh_st)
s3_4_acc_frameID_cknn_st = aux_f4.cKNN(s3_model_4, s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_frameID_tr_sh_st, s4_frameID_test_sh_st)
s3_5_acc_frameID_cknn_st = aux_f4.cKNN(s3_model_4, s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_frameID_tr_sh_st, s5_frameID_test_sh_st)

s4_3_acc_frameID_cknn_st = aux_f4.cKNN(s4_model_4, s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_frameID_tr_sh_st, s3_frameID_test_sh_st)
s4_4_acc_frameID_cknn_st = aux_f4.cKNN(s4_model_4, s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_frameID_tr_sh_st, s4_frameID_test_sh_st)
s4_5_acc_frameID_cknn_st = aux_f4.cKNN(s4_model_4, s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_frameID_tr_sh_st, s5_frameID_test_sh_st)

s5_3_acc_frameID_cknn_st = aux_f4.cKNN(s5_model_4, s3_neuraldata_tr_sh_st, s3_neuraldata_test_sh_st, s3_frameID_tr_sh_st, s3_frameID_test_sh_st)
s5_4_acc_frameID_cknn_st = aux_f4.cKNN(s5_model_4, s4_neuraldata_tr_sh_st, s4_neuraldata_test_sh_st, s4_frameID_tr_sh_st, s4_frameID_test_sh_st)
s5_5_acc_frameID_cknn_st = aux_f4.cKNN(s5_model_4, s5_neuraldata_tr_sh_st, s5_neuraldata_test_sh_st, s5_frameID_tr_sh_st, s5_frameID_test_sh_st)

accuracy_scores_frameID_cKNN_st = [s3_3_acc_frameID_cknn_st, s3_4_acc_frameID_cknn_st, s3_5_acc_frameID_cknn_st, s4_3_acc_frameID_cknn_st, s4_4_acc_frameID_cknn_st, s4_5_acc_frameID_cknn_st, s5_3_acc_frameID_cknn_st, s5_4_acc_frameID_cknn_st, s5_5_acc_frameID_cknn_st]

# 1. KNN v cKNN for filmID data - not stationary filtered 
aux_f4.plot_CKNN_KNN(accuracy_scores_filmID_KNN, accuracy_scores_filmID_cKNN, '- filmID - not stationary filtered (shuffled)')

# 2. KNN v cKNN for filmID data - stationary filtered
aux_f4.plot_CKNN_KNN(accuracy_scores_filmID_st_KNN, accuracy_scores_filmID_cKNN_st, '- filmID - stationary filtered (shuffled)')

# 3. KNN v cKNN for frameID data - not stationary filtered
aux_f4.plot_CKNN_KNN(accuracy_scores_frameID_KNN, accuracy_scores_frameID_cKNN, '- frameID - not stationary filtered (shuffled)')

# 4. KNN v cKNN for frameID data - stationary filtered
aux_f4.plot_CKNN_KNN(accuracy_scores_frameID_st_KNN, accuracy_scores_frameID_cKNN_st, '- frameID - stationary filtered (shuffled)')

#plt.show()
