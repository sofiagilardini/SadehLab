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

# KNN decoders for filmID data

KNN_decoder_s3 = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
KNN_decoder_s4 = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
KNN_decoder_s5 = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')

# train the KNN decoders

KNN_decoder_s3.fit(s3_neuraldata_tr, s3_filmID_tr)
KNN_decoder_s4.fit(s4_neuraldata_tr, s4_filmID_tr)
KNN_decoder_s5.fit(s5_neuraldata_tr, s5_filmID_tr)

# test the KNN decoders

s3_predicted_filmID = KNN_decoder_s3.predict(s3_neuraldata_test)
s4_predicted_filmID = KNN_decoder_s4.predict(s4_neuraldata_test)
s5_predicted_filmID = KNN_decoder_s5.predict(s5_neuraldata_test)

# get the accuracy scores

s3_3_acc = accuracy_score(s3_filmID_test, s3_predicted_filmID)
s4_4_acc = accuracy_score(s4_filmID_test, s4_predicted_filmID)
s5_5_acc = accuracy_score(s5_filmID_test, s5_predicted_filmID)

print('s3_3 KNN accuracy: ', s3_3_acc)
print('s4_4 KNN accuracy: ', s4_4_acc)
print('s5_4 KNN accuracy: ', s5_5_acc)

# cross check the accuracy scores 

s3_4_pred = KNN_decoder_s3.predict(s4_neuraldata_test)
s3_4_acc = accuracy_score(s4_filmID_test, s3_4_pred)
print('s3_4_acc', s3_4_acc)

s3_5_pred = KNN_decoder_s3.predict(s5_neuraldata_test)
s3_5_acc = accuracy_score(s5_filmID_test, s3_5_pred)
print('s3_5_acc', s3_5_acc)

s4_3_pred = KNN_decoder_s4.predict(s3_neuraldata_test)
s4_3_acc = accuracy_score(s3_filmID_test, s4_3_pred)
print('s4_3_acc', s4_3_acc)

s4_5_pred = KNN_decoder_s4.predict(s5_neuraldata_test)
s4_5_acc = accuracy_score(s5_filmID_test, s4_5_pred)
print('s4_5_acc', s4_5_acc)

s5_3_pred = KNN_decoder_s5.predict(s3_neuraldata_test)
s5_3_acc = accuracy_score(s3_filmID_test, s5_3_pred)
print('s5_3_acc', s5_3_acc)

s5_4_pred = KNN_decoder_s5.predict(s4_neuraldata_test)
s5_4_acc = accuracy_score(s4_filmID_test, s5_4_pred)
print('s5_4_acc', s5_4_acc)

accuracy_scores_filmID_KNN = [s3_3_acc, s3_4_acc, s3_5_acc, s4_3_acc, s4_4_acc, s4_5_acc, s5_3_acc, s5_4_acc, s5_5_acc]

# generate text file to save accuracy scores

# text_file_name = './information/KNN_perfomance/rawneural_filmID/KNN_raw.txt'

# with open(text_file_name, 'w') as f:
#     for item in accuracy_scores:
#         f.write("%s\n" % item)



# the above is KNN fitted on data which is unfiltered for stationary trials

# ---- repeated for stationary data comparison ------

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


# -------- KNN decoders on stationary data for filmID ----------

KNN_decoder_s3_st = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
KNN_decoder_s4_st = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
KNN_decoder_s5_st = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')

# train the KNN decoders

KNN_decoder_s3_st.fit(s3_neuraldata_tr_st, s3_filmID_tr_st)
KNN_decoder_s4_st.fit(s4_neuraldata_tr_st, s4_filmID_tr_st)
KNN_decoder_s5_st.fit(s5_neuraldata_tr_st, s5_filmID_tr_st)

# test the KNN decoders

s3_predicted_filmID_st = KNN_decoder_s3.predict(s3_neuraldata_test_st)
s4_predicted_filmID_st = KNN_decoder_s4.predict(s4_neuraldata_test_st)
s5_predicted_filmID_st = KNN_decoder_s5.predict(s5_neuraldata_test_st)

# cross check the KNN accuracy scores for stationary data 

# cross check the accuracy scores 

# KNN 3 tested on s3, s4, s5
s3_3_pred_st = KNN_decoder_s3_st.predict(s3_neuraldata_test_st)
s3_3_acc_st = accuracy_score(s3_filmID_test_st, s3_3_pred_st)
print('s3_3_acc_st', s3_3_acc_st)

s3_4_pred_st = KNN_decoder_s3_st.predict(s4_neuraldata_test_st)
s3_4_acc_st = accuracy_score(s4_filmID_test_st, s3_4_pred_st)
print('s3_4_acc_st', s3_4_acc_st)

s3_5_pred_st = KNN_decoder_s3_st.predict(s5_neuraldata_test_st)
s3_5_acc_st = accuracy_score(s5_filmID_test_st, s3_5_pred_st)
print('s3_5_acc_st', s3_5_acc_st)

# KNN 4 tested on s3, s4, s5
s4_3_pred_st = KNN_decoder_s4_st.predict(s3_neuraldata_test_st)
s4_3_acc_st = accuracy_score(s3_filmID_test_st, s4_3_pred_st)
print('s4_3_acc_st', s4_3_acc_st)

s4_4_pred_st = KNN_decoder_s4_st.predict(s4_neuraldata_test_st)
s4_4_acc_st = accuracy_score(s4_filmID_test_st, s4_4_pred_st)
print('s4_4_acc_st', s4_4_acc_st)

s4_5_pred_st = KNN_decoder_s4_st.predict(s5_neuraldata_test_st)
s4_5_acc_st = accuracy_score(s5_filmID_test_st, s4_5_pred_st)
print('s4_5_acc_st', s4_5_acc_st)

# KNN 5 tested on s3, s4, s5
s5_3_pred_st = KNN_decoder_s5_st.predict(s3_neuraldata_test_st)
s5_3_acc_st = accuracy_score(s3_filmID_test_st, s5_3_pred_st)
print('s5_3_acc_st', s4_3_acc_st)

s5_4_pred_st = KNN_decoder_s5_st.predict(s4_neuraldata_test_st)
s5_4_acc_st = accuracy_score(s4_filmID_test_st, s5_4_pred_st)
print('s5_4_acc_st', s5_4_acc_st)

s5_5_pred_st = KNN_decoder_s5_st.predict(s5_neuraldata_test_st)
s5_5_acc_st = accuracy_score(s5_filmID_test_st, s5_5_pred_st)
print('s5_5_acc_st', s5_5_acc_st)

accuracy_scores_filmID_st_KNN = [s3_3_acc_st, s3_4_acc_st, s3_5_acc_st, s4_3_acc_st, s4_4_acc_st, s4_5_acc_st, s5_3_acc_st, s5_4_acc_st, s5_5_acc_st]
#x_axis = np.arange(0, len(accuracy_scores), 1)
# plt.scatter(x_axis, accuracy_scores_st, c = 'r', label)
# plt.scatter(x_axis, accuracy_scores, c = 'b')

# save accuracy scores in text file 

# create a text file to save the accuracy scores


# text_file_name = './information/KNN_perfomance/neural_st_filmID/KNN_st.txt'

# with open(text_file_name, 'w') as f:
#     for item in accuracy_scores_st:
#         f.write("%s\n" % item)


# plt.plot(x_axis, accuracy_scores, 'x' , color = 'b', label = 'raw data')
# plt.plot(x_axis, accuracy_scores_st, 'x' , color = 'r', label = 'stationary filtered data')
# plt.title('KNN performance on neural data for filmID - no Cebra')
# plt.xlabel('s3_3, s3_4, s3_5, s4_3, s4_4...etc.')
# plt.legend()
# plt.ylim(0,1)
# plt.savefig('./information/figures/KNN_filmID.png')
# #plt.show()


# make KNN stationary and raw the same number of data points !! --- !!! --- !!1 
# the performance may be different simply because raw data has more data ? 


## ------ train cebra -------------

# define cebra model 


single_cebra_model = cebra.CEBRA(
    model_architecture = "offset10-model",
    batch_size = 516, #??
    temperature_mode="auto",
    learning_rate = 0.001,
    max_iterations = 6000,
    max_adapt_iterations = 300,
    time_offsets = 10,
    output_dimension = 12, # this is the change specific to this script (otherise is 3)
    device = device,
    verbose = True,
    conditional = 'time_delta',
    # hybrid = True is the only line that is different from models4.py
)

# create deepcopy for initialising 12 models

models = []
for i in range(12):
    models.append(deepcopy(single_cebra_model))

# # train the models for non-stationary data for FilmID (./training_results/models_14)
# aux_f4.trainCebraBehaviour(models[0], s3_neuraldata_tr, s3_neuraldata_test, s3_filmID_tr, s3_filmID_test, 3, 'm14_s3', s3_indices_tr, s3_indices_test, 'models_14')
# aux_f4.trainCebraBehaviour(models[1], s4_neuraldata_tr, s4_neuraldata_test, s4_filmID_tr, s4_filmID_test, 4, 'm14_s4', s4_indices_tr, s4_indices_test, 'models_14')
# aux_f4.trainCebraBehaviour(models[2], s5_neuraldata_tr, s5_neuraldata_test, s5_filmID_tr, s5_filmID_test, 5, 'm14_s5', s5_indices_tr, s5_indices_test, 'models_14')

# # train the models for stationary data for FilmID (./training_results/models_15)

# aux_f4.trainCebraBehaviour(models[3], s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_filmID_tr_st, s3_filmID_test_st, 3, 'm15_s3', s3_indices_tr_st, s3_indices_test_st, 'models_15')
# aux_f4.trainCebraBehaviour(models[4], s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_filmID_tr_st, s4_filmID_test_st, 4, 'm15_s4', s4_indices_tr_st, s4_indices_test_st, 'models_15')
# aux_f4.trainCebraBehaviour(models[5], s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_filmID_tr_st, s5_filmID_test_st, 5, 'm15_s5', s5_indices_tr_st, s5_indices_test_st, 'models_15')    

###----- end of FilmID ------ ### 

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

# # train Cebra models on frameID data

# # train the models for non-stationary data for FrameID (./training_results/models_16) -- DONE

# aux_f4.trainCebraBehaviour(models[6], s3_neuraldata_tr, s3_neuraldata_test, s3_frameID_tr, s3_frameID_test, 3, 'm16_s3', s3_indices_tr, s3_indices_test, 'models_16')
# aux_f4.trainCebraBehaviour(models[7], s4_neuraldata_tr, s4_neuraldata_test, s4_frameID_tr, s4_frameID_test, 4, 'm16_s4', s4_indices_tr, s4_indices_test, 'models_16')
# aux_f4.trainCebraBehaviour(models[8], s5_neuraldata_tr, s5_neuraldata_test, s5_frameID_tr, s5_frameID_test, 5, 'm16_s5', s5_indices_tr, s5_indices_test, 'models_16')

# # train the models for stationary data for FrameID (./training_results/models_17) -- DONE

# aux_f4.trainCebraBehaviour(models[9], s3_neuraldata_tr_st, s3_neuraldata_test_st, s3_frameID_tr_st, s3_frameID_test_st, 3, 'm17_s3', s3_indices_tr_st, s3_indices_test_st, 'models_17')
# aux_f4.trainCebraBehaviour(models[10], s4_neuraldata_tr_st, s4_neuraldata_test_st, s4_frameID_tr_st, s4_frameID_test_st, 4, 'm17_s4', s4_indices_tr_st, s4_indices_test_st, 'models_17')
# aux_f4.trainCebraBehaviour(models[11], s5_neuraldata_tr_st, s5_neuraldata_test_st, s5_frameID_tr_st, s5_frameID_test_st, 5, 'm17_s5', s5_indices_tr_st, s5_indices_test_st, 'models_17')

## --- KNN --- ##

# check KNN decoder performance on frameID data (not filtered for stationary trials)

# KNN decoders for frameID data

KNN_decoder_s3_frameID = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
KNN_decoder_s4_frameID = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
KNN_decoder_s5_frameID = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')

# fit the KNN decoders

KNN_decoder_s3_frameID.fit(s3_neuraldata_tr, s3_frameID_tr)
KNN_decoder_s4_frameID.fit(s4_neuraldata_tr, s4_frameID_tr)
KNN_decoder_s5_frameID.fit(s5_neuraldata_tr, s5_frameID_tr)

# cross - test the KNN decoders for non stationary filtered data

s3_3_pred_frameID = KNN_decoder_s3_frameID.predict(s3_neuraldata_test)
s3_3_acc_frameID = accuracy_score(s3_frameID_test, s3_3_pred_frameID)
print('s3_3_acc_frameID', s3_3_acc_frameID)

s3_4_pred_frameID = KNN_decoder_s3_frameID.predict(s4_neuraldata_test)
s3_4_acc_frameID = accuracy_score(s4_frameID_test, s3_4_pred_frameID)
print('s3_4_acc_frameID', s3_4_acc_frameID)

s3_5_pred_frameID = KNN_decoder_s3_frameID.predict(s5_neuraldata_test)
s3_5_acc_frameID = accuracy_score(s5_frameID_test, s3_5_pred_frameID)
print('s3_5_acc_frameID', s3_5_acc_frameID)

s4_3_pred_frameID = KNN_decoder_s4_frameID.predict(s3_neuraldata_test)
s4_3_acc_frameID = accuracy_score(s3_frameID_test, s4_3_pred_frameID)
print('s4_3_acc_frameID', s4_3_acc_frameID)

s4_4_pred_frameID = KNN_decoder_s4_frameID.predict(s4_neuraldata_test)
s4_4_acc_frameID = accuracy_score(s4_frameID_test, s4_4_pred_frameID)
print('s4_4_acc_frameID', s4_4_acc_frameID)

s4_5_pred_frameID = KNN_decoder_s4_frameID.predict(s5_neuraldata_test)
s4_5_acc_frameID = accuracy_score(s5_frameID_test, s4_5_pred_frameID)
print('s4_5_acc_frameID', s4_5_acc_frameID)

s5_3_pred_frameID = KNN_decoder_s5_frameID.predict(s3_neuraldata_test)
s5_3_acc_frameID = accuracy_score(s3_frameID_test, s5_3_pred_frameID)
print('s5_3_acc_frameID', s5_3_acc_frameID)

s5_4_pred_frameID = KNN_decoder_s5_frameID.predict(s4_neuraldata_test)
s5_4_acc_frameID = accuracy_score(s4_frameID_test, s5_4_pred_frameID)
print('s5_4_acc_frameID', s5_4_acc_frameID)

s5_5_pred_frameID = KNN_decoder_s5_frameID.predict(s5_neuraldata_test)
s5_5_acc_frameID = accuracy_score(s5_frameID_test, s5_5_pred_frameID)
print('s5_5_acc_frameID', s5_5_acc_frameID)

accuracy_scores_frameID_KNN = [s3_3_acc_frameID, s3_4_acc_frameID, s3_5_acc_frameID, s4_3_acc_frameID, s4_4_acc_frameID, s4_5_acc_frameID, s5_3_acc_frameID, s5_4_acc_frameID, s5_5_acc_frameID]

# save accuracy scores in text file

# with open('./information/KNN_perfomance/rawneural_frameID/KNN_raw.txt', 'w') as f:
#     for item in accuracy_scores_frameID:
#         f.write("%s\n" % item)


# check KNN decoder performance on frameID data (filtered for stationary trials)

# KNN decoders for frameID data

KNN_decoder_s3_frameID_st = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
KNN_decoder_s4_frameID_st = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
KNN_decoder_s5_frameID_st = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')

# fit the KNN decoders

KNN_decoder_s3_frameID_st.fit(s3_neuraldata_tr_st, s3_frameID_tr_st)
KNN_decoder_s4_frameID_st.fit(s4_neuraldata_tr_st, s4_frameID_tr_st)
KNN_decoder_s5_frameID_st.fit(s5_neuraldata_tr_st, s5_frameID_tr_st)

# cross - test the KNN decoders for non stationary filtered data

s3_3_pred_frameID_st = KNN_decoder_s3_frameID_st.predict(s3_neuraldata_test_st)
s3_3_acc_frameID_st = accuracy_score(s3_frameID_test_st, s3_3_pred_frameID_st)

s3_4_pred_frameID_st = KNN_decoder_s3_frameID_st.predict(s4_neuraldata_test_st)
s3_4_acc_frameID_st = accuracy_score(s4_frameID_test_st, s3_4_pred_frameID_st)

s3_5_pred_frameID_st = KNN_decoder_s3_frameID_st.predict(s5_neuraldata_test_st)
s3_5_acc_frameID_st = accuracy_score(s5_frameID_test_st, s3_5_pred_frameID_st)

s4_3_pred_frameID_st = KNN_decoder_s4_frameID_st.predict(s3_neuraldata_test_st)
s4_3_acc_frameID_st = accuracy_score(s3_frameID_test_st, s4_3_pred_frameID_st)

s4_4_pred_frameID_st = KNN_decoder_s4_frameID_st.predict(s4_neuraldata_test_st)
s4_4_acc_frameID_st = accuracy_score(s4_frameID_test_st, s4_4_pred_frameID_st)

s4_5_pred_frameID_st = KNN_decoder_s4_frameID_st.predict(s5_neuraldata_test_st)
s4_5_acc_frameID_st = accuracy_score(s5_frameID_test_st, s4_5_pred_frameID_st)

s5_3_pred_frameID_st = KNN_decoder_s5_frameID_st.predict(s3_neuraldata_test_st)
s5_3_acc_frameID_st = accuracy_score(s3_frameID_test_st, s5_3_pred_frameID_st)

s5_4_pred_frameID_st = KNN_decoder_s5_frameID_st.predict(s4_neuraldata_test_st)
s5_4_acc_frameID_st = accuracy_score(s4_frameID_test_st, s5_4_pred_frameID_st)

s5_5_pred_frameID_st = KNN_decoder_s5_frameID_st.predict(s5_neuraldata_test_st)
s5_5_acc_frameID_st = accuracy_score(s5_frameID_test_st, s5_5_pred_frameID_st)

accuracy_scores_frameID_st_KNN = [s3_3_acc_frameID_st, s3_4_acc_frameID_st, s3_5_acc_frameID_st, s4_3_acc_frameID_st, s4_4_acc_frameID_st, s4_5_acc_frameID_st, s5_3_acc_frameID_st, s5_4_acc_frameID_st, s5_5_acc_frameID_st]

# save accuracy scores in text file

# with open('./information/KNN_perfomance/neural_st_frameID/KNN_st.txt', 'w') as f:
#     for item in accuracy_scores_frameID_st:
#         f.write("%s\n" % item)


# plot the accuracy scores for frameID data

# plt.plot(x_axis, accuracy_scores_frameID, 'x' , color = 'g', label = 'raw data_frameID')
# plt.plot(x_axis, accuracy_scores_frameID_st, 'x' , color = 'y', label = 'stationary filtered data_frameID')
# plt.title('KNN performance on neural data for frameID - no Cebra')
# plt.savefig('./information/figures/KNN_frameID.png')
# #plt.show()


### --- begin testing cKNN --- ###
# cKNN is a KNN decoder which is trained on the cebra model output 

# begin with filmID data - load models and generate embeddings 

# load the models for raw data (filmID)

s3_model_5 = CEBRA.load('./training_results/models_14/cebra_model_m14_s3.pt', map_location=torch.device("cpu"))
s4_model_5 = CEBRA.load('./training_results/models_14/cebra_model_m14_s4.pt', map_location=torch.device("cpu"))
s5_model_5 = CEBRA.load('./training_results/models_14/cebra_model_m14_s5.pt', map_location=torch.device("cpu"))

# load the models for stationary data (filmID)
s3_model_6 = CEBRA.load('./training_results/models_15/cebra_model_m15_s3.pt', map_location=torch.device("cpu"))
s4_model_6 = CEBRA.load('./training_results/models_15/cebra_model_m15_s4.pt', map_location=torch.device("cpu"))
s5_model_6 = CEBRA.load('./training_results/models_15/cebra_model_m15_s5.pt', map_location=torch.device("cpu"))

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

s3_model_7 = CEBRA.load('./training_results/models_16/cebra_model_m16_s3.pt', map_location=torch.device("cpu"))
s4_model_7 = CEBRA.load('./training_results/models_16/cebra_model_m16_s4.pt', map_location=torch.device("cpu"))
s5_model_7 = CEBRA.load('./training_results/models_16/cebra_model_m16_s5.pt', map_location=torch.device("cpu"))

# load the models for stationary data (frameID)

s3_model_8 = CEBRA.load('./training_results/models_17/cebra_model_m17_s3.pt', map_location=torch.device("cpu"))
s4_model_8 = CEBRA.load('./training_results/models_17/cebra_model_m17_s4.pt', map_location=torch.device("cpu"))
s5_model_8 = CEBRA.load('./training_results/models_17/cebra_model_m17_s5.pt', map_location=torch.device("cpu"))

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

# # 1. KNN v cKNN for filmID data - not stationary filtered 
# aux_f4.plot_CKNN_KNN(accuracy_scores_filmID_KNN, accuracy_scores_filmID_cKNN, '- filmID - not stationary filtered')

# # 2. KNN v cKNN for filmID data - stationary filtered
# aux_f4.plot_CKNN_KNN(accuracy_scores_filmID_st_KNN, accuracy_scores_filmID_cKNN_st, '- filmID - stationary filtered')

# # 3. KNN v cKNN for frameID data - not stationary filtered
# aux_f4.plot_CKNN_KNN(accuracy_scores_frameID_KNN, accuracy_scores_frameID_cKNN, '- frameID - not stationary filtered')

# # 4. KNN v cKNN for frameID data - stationary filtered
# aux_f4.plot_CKNN_KNN(accuracy_scores_frameID_st_KNN, accuracy_scores_frameID_cKNN_st, '- frameID - stationary filtered')

#plt.show()

