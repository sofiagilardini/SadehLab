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
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#import functions2 as aux_f2
import functions4 as aux_f4

# decide Animal, type of film, and red/green

chosenAnimal = "A118"
A118 = classes.A118

Film = "Compilation"
Colour = "red"
iterations = 9000

paths_neuraldata = aux_f4.getneuraldata(chosenAnimal, Film, Colour, False)

# choose one session to train and test on

# paths neural data should always be a list of length 5 (5 sessions)
# session 2 and 6 have 6 stimID - index3 will not be list of  0's

# session 3, 4, 5 have 1 stimID ---- focus on this first -- index3 will be list of 0's BUT 
# would otherwise not be if we were to use session 2 or 6

print('paths neural data', paths_neuraldata)

for i in range(len(paths_neuraldata)):
    if 'S3' in paths_neuraldata[i]:
        path_session3  = paths_neuraldata[i]
        print('path session 3', path_session3)
    elif 'S4' in paths_neuraldata[i]:
        path_session4 = paths_neuraldata[i]
        print('path session 4', path_session4)

    elif 'S5' in paths_neuraldata[i]:
        path_session5 = paths_neuraldata[i]
        print('path session 5', path_session5)

# # removing this because of error with lambda function

# path_session3 = paths_neuraldata[1]
# path_session4 = paths_neuraldata[2]
# path_session5 = paths_neuraldata[3]


if 'S3' not in path_session3:
    print('error: session 3 path not correct')


if 'S4' not in path_session4:
    print('error: session 4 path not correct')

if 'S5' not in path_session5:
    print('error: session 5 path not correct')
    

# neural data should be a 3D array of shape (ROI x frames x trials)

neuraldata_session3 = np.load(path_session3, allow_pickle=True)
neuraldata_session4 = np.load(path_session4, allow_pickle=True)
neuraldata_session5 = np.load(path_session5, allow_pickle=True)

# ----------------- NOT STATIONARY TRIALS ----------------------

#print(neuraldata_session3.shape)
# 561 x 113 x 100 x 1

# number of trials for each session is not equal. S3: 100, S4: 140, S5: 60

# split train and test 80-20

s3_numtrials = neuraldata_session3.shape[2]
s4_numtrials = neuraldata_session4.shape[2]
s5_numtrials = neuraldata_session5.shape[2]

# s3_indices_tr = random.sample(range(s3_numtrials), int(s3_numtrials*0.8))
# s3_indices_test = [x for x in range(s3_numtrials) if x not in s3_indices_tr]

# s4_indices_tr = random.sample(range(s4_numtrials), int(s4_numtrials*0.8))
# s4_indices_test = [x for x in range(s4_numtrials) if x not in s4_indices_tr]

# s5_indices_tr = random.sample(range(s5_numtrials), int(s5_numtrials*0.8))
# s5_indices_test = [x for x in range(s5_numtrials) if x not in s5_indices_tr]

# # save these indices to information folder -- SAVED - DO NOT RERUN

# np.save('./information/indices/A118_s3_indices_tr.npy', s3_indices_tr)
# np.save('./information/indices/A118_s3_indices_test.npy', s3_indices_test)

# np.save('./information/indices/A118_s4_indices_tr.npy', s4_indices_tr)
# np.save('./information/indices/A118_s4_indices_test.npy', s4_indices_test)

# np.save('./information/indices/A118_s5_indices_tr.npy', s5_indices_tr)
# np.save('./information/indices/A118_s5_indices_test.npy', s5_indices_test)

# load these indices from information folder

s3_indices_tr = np.load('./information/indices/A118_s3_indices_tr.npy')
s3_indices_test = np.load('./information/indices/A118_s3_indices_test.npy')

s4_indices_tr = np.load('./information/indices/A118_s4_indices_tr.npy')
s4_indices_test = np.load('./information/indices/A118_s4_indices_test.npy')

s5_indices_tr = np.load('./information/indices/A118_s5_indices_tr.npy')
s5_indices_test = np.load('./information/indices/A118_s5_indices_test.npy')

# concatenate the neural data for training and testing into a 2D array

neuraldata_session3_tr = aux_f4.concatenate_neuraldata(neuraldata_session3, s3_indices_tr, 'A118')

neuraldata_session3_test = aux_f4.concatenate_neuraldata(neuraldata_session3, s3_indices_test, 'A118')

neuraldata_session4_tr = aux_f4.concatenate_neuraldata(neuraldata_session4, s4_indices_tr, 'A118')
neuraldata_session4_test = aux_f4.concatenate_neuraldata(neuraldata_session4, s4_indices_test, 'A118')

neuraldata_session5_tr = aux_f4.concatenate_neuraldata(neuraldata_session5, s5_indices_tr, 'A118')
neuraldata_session5_test = aux_f4.concatenate_neuraldata(neuraldata_session5, s5_indices_test, 'A118')

# # save these concatenated arrays to information folder / neuraldata -- SAVED -- DO NOT RERUN

# np.save('./information/neuraldata/A118_neuraldata_session3_tr.npy', neuraldata_session3_tr)
# np.save('./information/neuraldata/A118_neuraldata_session3_test.npy', neuraldata_session3_test)

# np.save('./information/neuraldata/A118_neuraldata_session4_tr.npy', neuraldata_session4_tr)
# np.save('./information/neuraldata/A118_neuraldata_session4_test.npy', neuraldata_session4_test)

# np.save('./information/neuraldata/A118_neuraldata_session5_tr.npy', neuraldata_session5_tr)
# np.save('./information/neuraldata/A118_neuraldata_session5_test.npy', neuraldata_session5_test)

# load these concatenated arrays from information folder / neuraldata

neuraldata_session3_tr = np.load('./information/neuraldata/A118_neuraldata_session3_tr.npy')
neuraldata_session3_test = np.load('./information/neuraldata/A118_neuraldata_session3_test.npy')

neuraldata_session4_tr = np.load('./information/neuraldata/A118_neuraldata_session4_tr.npy')
neuraldata_session4_test = np.load('./information/neuraldata/A118_neuraldata_session4_test.npy')

neuraldata_session5_tr = np.load('./information/neuraldata/A118_neuraldata_session5_tr.npy')
neuraldata_session5_test = np.load('./information/neuraldata/A118_neuraldata_session5_test.npy')

# get FilmID for these indices 

s3_filmID = aux_f4.compilationSorterFilm(path_session3,chosenAnimal, neuraldata_session3.shape[2], 3)
s4_filmID = aux_f4.compilationSorterFilm(path_session4,chosenAnimal, neuraldata_session4.shape[2], 4)
s5_filmID = aux_f4.compilationSorterFilm(path_session5,chosenAnimal, neuraldata_session5.shape[2], 5)

s3_filmID_tr = s3_filmID[:, s3_indices_tr]
s3_filmID_test = s3_filmID[:, s3_indices_test]

s4_filmID_tr = s4_filmID[:, s4_indices_tr]
s4_filmID_test = s4_filmID[:, s4_indices_test]

s5_filmID_tr = s5_filmID[:, s5_indices_tr]
s5_filmID_test = s5_filmID[:, s5_indices_test]


# get FrameID for these indices 

s3_frameID = aux_f4.compilationSorterFrames(path_session3,chosenAnimal)
s4_frameID = aux_f4.compilationSorterFrames(path_session4,chosenAnimal)
s5_frameID = aux_f4.compilationSorterFrames(path_session5,chosenAnimal)

print('s3 frame ID shape', s3_frameID.shape)
print('trials tr indices', s3_indices_tr)

print('s4 frame ID shape', s4_frameID.shape)
print('trials tr indices', s4_indices_tr)

s3_frameID_tr = s3_frameID[:, s3_indices_tr]
s3_frameID_test = s3_frameID[:, s3_indices_test]

s4_frameID_tr = s4_frameID[:, s4_indices_tr]
s4_frameID_test = s4_frameID[:, s4_indices_test]

s5_frameID_tr = s5_frameID[:, s5_indices_tr]
s5_frameID_test = s5_frameID[:, s5_indices_test]


# concatenate FilmID into a 1D array

s3_filmID_tr = s3_filmID_tr.flatten('F')
s3_filmID_test = s3_filmID_test.flatten('F')
s4_filmID_tr = s4_filmID_tr.flatten('F')
s4_filmID_test = s4_filmID_test.flatten('F')
s5_filmID_tr = s5_filmID_tr.flatten('F')
s5_filmID_test = s5_filmID_test.flatten('F')

# concatenate FrameID into a 1D array

s3_frameID_tr = s3_frameID_tr.flatten('F')
s3_frameID_test = s3_frameID_test.flatten('F')
s4_frameID_tr = s4_frameID_tr.flatten('F')
s4_frameID_test = s4_frameID_test.flatten('F')
s5_frameID_tr = s5_frameID_tr.flatten('F')
s5_frameID_test = s5_frameID_test.flatten('F')


# # save filmID's to information folder / filmID --- SAVED - DO NOT RERUN

# np.save('./information/filmID/A118_s3_filmID_tr.npy', s3_filmID_tr)
# np.save('./information/filmID/A118_s3_filmID_test.npy', s3_filmID_test)
# np.save('./information/filmID/A118_s4_filmID_tr.npy', s4_filmID_tr)
# np.save('./information/filmID/A118_s4_filmID_test.npy', s4_filmID_test)
# np.save('./information/filmID/A118_s5_filmID_tr.npy', s5_filmID_tr)
# np.save('./information/filmID/A118_s5_filmID_test.npy', s5_filmID_test)

# load 

s3_filmID_tr = np.load('./information/filmID/A118_s3_filmID_tr.npy')
s3_filmID_test = np.load('./information/filmID/A118_s3_filmID_test.npy')
s4_filmID_tr = np.load('./information/filmID/A118_s4_filmID_tr.npy')
s4_filmID_test = np.load('./information/filmID/A118_s4_filmID_test.npy')
s5_filmID_tr = np.load('./information/filmID/A118_s5_filmID_tr.npy')
s5_filmID_test = np.load('./information/filmID/A118_s5_filmID_test.npy')


# # save frameID's to information folder / frameID

# np.save('./information/frameID/A118_s3_frameID_tr.npy', s3_frameID_tr)
# np.save('./information/frameID/A118_s3_frameID_test.npy', s3_frameID_test)
# np.save('./information/frameID/A118_s4_frameID_tr.npy', s4_frameID_tr)
# np.save('./information/frameID/A118_s4_frameID_test.npy', s4_frameID_test)
# np.save('./information/frameID/A118_s5_frameID_tr.npy', s5_frameID_tr)
# np.save('./information/frameID/A118_s5_frameID_test.npy', s5_frameID_test)

# load 

s3_frameID_tr = np.load('./information/frameID/A118_s3_frameID_tr.npy')
s3_frameID_test = np.load('./information/frameID/A118_s3_frameID_test.npy')
s4_frameID_tr = np.load('./information/frameID/A118_s4_frameID_tr.npy')
s4_frameID_test = np.load('./information/frameID/A118_s4_frameID_test.npy')
s5_frameID_tr = np.load('./information/frameID/A118_s5_frameID_tr.npy')
s5_frameID_test = np.load('./information/frameID/A118_s5_frameID_test.npy')


# repeat all but for data stationary-filtered 

# find the stationary trials by using 'encoder' data in behaviour

s3_stationarytrials = aux_f4.getStationaryIndex(path_session3)
s4_stationarytrials = aux_f4.getStationaryIndex(path_session4)
s5_stationarytrials = aux_f4.getStationaryIndex(path_session5)

# filter the training and testing indices according to the stationary trials -- done

s3_indices_tr_st = [x for x in s3_indices_tr if x in s3_stationarytrials]
s3_indices_test_st = [x for x in s3_indices_test if x in s3_stationarytrials]

s4_indices_tr_st = [x for x in s4_indices_tr if x in s4_stationarytrials]
s4_indices_test_st = [x for x in s4_indices_test if x in s4_stationarytrials]

s5_indices_tr_st = [x for x in s5_indices_tr if x in s5_stationarytrials]
s5_indices_test_st = [x for x in s5_indices_test if x in s5_stationarytrials]

## save these indices to information folder -- SAVED - DO NOT RERUN

# np.save('./information/indices_stationary/A118_s3_tr_st', s3_indices_tr_st)
# np.save('./information/indices_stationary/A118_s3_test_st', s3_indices_test_st)
# np.save('./information/indices_stationary/A118_s4_tr_st', s4_indices_tr_st)
# np.save('./information/indices_stationary/A118_s4_test_st', s4_indices_test_st)
# np.save('./information/indices_stationary/A118_s5_tr_st', s5_indices_tr_st)
# np.save('./information/indices_stationary/A118_s5_test_st', s5_indices_test_st)

# # load these indices from information folder

s3_indices_tr_st = np.load('./information/indices_stationary/A118_s3_tr_st.npy')
s3_indices_test_st = np.load('./information/indices_stationary/A118_s3_test_st.npy')
s4_indices_tr_st = np.load('./information/indices_stationary/A118_s4_tr_st.npy')
s4_indices_test_st = np.load('./information/indices_stationary/A118_s4_test_st.npy')
s5_indices_tr_st = np.load('./information/indices_stationary/A118_s5_tr_st.npy')
s5_indices_test_st = np.load('./information/indices_stationary/A118_s5_test_st.npy')

# neural data filtered for stationary trials  -- done

neuraldata_session3_tr_st = aux_f4.concatenate_neuraldata(neuraldata_session3, s3_indices_tr_st, 'A118')
neuraldata_session3_test_st = aux_f4.concatenate_neuraldata(neuraldata_session3, s3_indices_test_st, 'A118')

neuraldata_session4_tr_st = aux_f4.concatenate_neuraldata(neuraldata_session4, s4_indices_tr_st, 'A118')
neuraldata_session4_test_st = aux_f4.concatenate_neuraldata(neuraldata_session4, s4_indices_test_st, 'A118')

neuraldata_session5_tr_st = aux_f4.concatenate_neuraldata(neuraldata_session5, s5_indices_tr_st, 'A118')
neuraldata_session5_test_st = aux_f4.concatenate_neuraldata(neuraldata_session5, s5_indices_test_st, 'A118')

## SAVED - DO NOT RERUN

# np.save('./information/neuraldata_stationary/A118_neuraldata_s3_tr_st.npy', neuraldata_session3_tr_st)
# np.save('./information/neuraldata_stationary/A118_neuraldata_s3_test_st.npy', neuraldata_session3_test_st)

# np.save('./information/neuraldata_stationary/A118_neuraldata_s4_tr_st.npy', neuraldata_session4_tr_st)
# np.save('./information/neuraldata_stationary/A118_neuraldata_s4_test_st.npy', neuraldata_session4_test_st)

# np.save('./information/neuraldata_stationary/A118_neuraldata_s5_tr_st.npy', neuraldata_session5_tr_st)
# np.save('./information/neuraldata_stationary/A118_neuraldata_s5_test_st.npy', neuraldata_session5_test_st)

# load these concatenated arrays from information folder / neuraldata

neuraldata_session3_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s3_tr_st.npy')
neuraldata_session3_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s3_test_st.npy')
neuraldata_session4_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s4_tr_st.npy')
neuraldata_session4_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s4_test_st.npy')
neuraldata_session5_tr_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s5_tr_st.npy')
neuraldata_session5_test_st = np.load('./information/neuraldata_stationary/A118_neuraldata_s5_test_st.npy')


# filmID stationary trials -- done

s3_filmID_tr_st = s3_filmID[:, s3_indices_tr_st].flatten('F')
s3_filmID_test_st = s3_filmID[:, s3_indices_test_st].flatten('F')

s4_filmID_tr_st = s4_filmID[:, s4_indices_tr_st].flatten('F')
s4_filmID_test_st = s4_filmID[:, s4_indices_test_st].flatten('F')

s5_filmID_tr_st = s5_filmID[:, s5_indices_tr_st].flatten('F')
s5_filmID_test_st = s5_filmID[:, s5_indices_test_st].flatten('F')

# ## SAVED - DO NOT RERUN

# np.save('./information/filmID_stationary/A118_s3_filmID_tr_st.npy', s3_filmID_tr_st)
# np.save('./information/filmID_stationary/A118_s3_filmID_test_st.npy', s3_filmID_test_st)
# np.save('./information/filmID_stationary/A118_s4_filmID_tr_st.npy', s4_filmID_tr_st)
# np.save('./information/filmID_stationary/A118_s4_filmID_test_st.npy', s4_filmID_test_st)
# np.save('./information/filmID_stationary/A118_s5_filmID_tr_st.npy', s5_filmID_tr_st)
# np.save('./information/filmID_stationary/A118_s5_filmID_test_st.npy', s5_filmID_test_st)

# frameID stationary trials

s3_frameID_tr_st = s3_frameID[:, s3_indices_tr_st].flatten('F')
s3_frameID_test_st = s3_frameID[:, s3_indices_test_st].flatten('F')
s4_frameID_tr_st = s4_frameID[:, s4_indices_tr_st].flatten('F')
s4_frameID_test_st = s4_frameID[:, s4_indices_test_st].flatten('F')
s5_frameID_tr_st = s5_frameID[:, s5_indices_tr_st].flatten('F')
s5_frameID_test_st = s5_frameID[:, s5_indices_test_st].flatten('F')

# # save frameID's to information folder / frameID_stationary  -- SAVED / DO NOT RE-RUN

# np.save('./information/frameID_stationary/A118_s3_frameID_tr_st.npy', s3_frameID_tr_st)
# np.save('./information/frameID_stationary/A118_s3_frameID_test_st.npy', s3_frameID_test_st)
# np.save('./information/frameID_stationary/A118_s4_frameID_tr_st.npy', s4_frameID_tr_st)
# np.save('./information/frameID_stationary/A118_s4_frameID_test_st.npy', s4_frameID_test_st)
# np.save('./information/frameID_stationary/A118_s5_frameID_tr_st.npy', s5_frameID_tr_st)
# np.save('./information/frameID_stationary/A118_s5_frameID_test_st.npy', s5_frameID_test_st)


# ------------------ pupilDiam data ---------------

# get pupilDiam data (all)

s3_pupilDiam = aux_f4.getBehaviourData(path_session3, 'pupilDiam')
s4_pupilDiam = aux_f4.getBehaviourData(path_session4, 'pupilDiam')
s5_pupilDiam = aux_f4.getBehaviourData(path_session5, 'pupilDiam')



# get pupilDiam data for raw data 

s3_pupilDiam_tr = aux_f4.concatenateData(s3_pupilDiam, s3_indices_tr, "A118")
s3_pupilDiam_test = aux_f4.concatenateData(s3_pupilDiam, s3_indices_test, "A118")
s4_pupilDiam_tr = aux_f4.concatenateData(s4_pupilDiam, s4_indices_tr, "A118")
s4_pupilDiam_test = aux_f4.concatenateData(s4_pupilDiam, s4_indices_test, "A118")
s5_pupilDiam_tr = aux_f4.concatenateData(s5_pupilDiam, s5_indices_tr, "A118")
s5_pupilDiam_test = aux_f4.concatenateData(s5_pupilDiam, s5_indices_test, "A118")


# flatten for training compatibility

s3_pupilDiam_tr = s3_pupilDiam_tr.flatten('F')
s3_pupilDiam_test = s3_pupilDiam_test.flatten('F')
s4_pupilDiam_tr = s4_pupilDiam_tr.flatten('F')
s4_pupilDiam_test = s4_pupilDiam_test.flatten('F')
s5_pupilDiam_tr = s5_pupilDiam_tr.flatten('F')
s5_pupilDiam_test = s5_pupilDiam_test.flatten('F')

# save pupilDiam's to information folder / pupilDiam -- saved! 

np.save('./information/pupilDiam/A118_s3_pupilDiam_tr.npy', s3_pupilDiam_tr)
np.save('./information/pupilDiam/A118_s3_pupilDiam_test.npy', s3_pupilDiam_test)
np.save('./information/pupilDiam/A118_s4_pupilDiam_tr.npy', s4_pupilDiam_tr)
np.save('./information/pupilDiam/A118_s4_pupilDiam_test.npy', s4_pupilDiam_test)
np.save('./information/pupilDiam/A118_s5_pupilDiam_tr.npy', s5_pupilDiam_tr)
np.save('./information/pupilDiam/A118_s5_pupilDiam_test.npy', s5_pupilDiam_test)

# get pupilDiam data for stationary trials

s3_pupilDiam_tr_st = aux_f4.concatenateData(s3_pupilDiam, s3_indices_tr_st, "A118")
s3_pupilDiam_test_st = aux_f4.concatenateData(s3_pupilDiam, s3_indices_test_st, "A118")
s4_pupilDiam_tr_st = aux_f4.concatenateData(s4_pupilDiam, s4_indices_tr_st, "A118")
s4_pupilDiam_test_st = aux_f4.concatenateData(s4_pupilDiam, s4_indices_test_st, "A118")
s5_pupilDiam_tr_st = aux_f4.concatenateData(s5_pupilDiam, s5_indices_tr_st, "A118")
s5_pupilDiam_test_st = aux_f4.concatenateData(s5_pupilDiam, s5_indices_test_st, "A118")

# flatten for training compatibility

s3_pupilDiam_tr_st = s3_pupilDiam_tr_st.flatten('F')
s3_pupilDiam_test_st = s3_pupilDiam_test_st.flatten('F')
s4_pupilDiam_tr_st = s4_pupilDiam_tr_st.flatten('F')
s4_pupilDiam_test_st = s4_pupilDiam_test_st.flatten('F')
s5_pupilDiam_tr_st = s5_pupilDiam_tr_st.flatten('F')
s5_pupilDiam_test_st = s5_pupilDiam_test_st.flatten('F')


# save pupilDiam's to information folder / pupilDiam_st -- saved!

np.save('./information/pupilDiam_st/A118_s3_pupilDiam_tr_st.npy', s3_pupilDiam_tr_st)
np.save('./information/pupilDiam_st/A118_s3_pupilDiam_test_st.npy', s3_pupilDiam_test_st)
np.save('./information/pupilDiam_st/A118_s4_pupilDiam_tr_st.npy', s4_pupilDiam_tr_st)
np.save('./information/pupilDiam_st/A118_s4_pupilDiam_test_st.npy', s4_pupilDiam_test_st)
np.save('./information/pupilDiam_st/A118_s5_pupilDiam_tr_st.npy', s5_pupilDiam_tr_st)
np.save('./information/pupilDiam_st/A118_s5_pupilDiam_test_st.npy', s5_pupilDiam_test_st)



## --------------------------------- shuffle data ---------------------------- ##

# --- for data not stationary filtered ---- #

# shuffle the indices for training and testing



#neuraldata_list = [neuraldata_session3_tr, neuraldata_session3_test, neuraldata_session4_tr, neuraldata_session4_test, neuraldata_session5_tr, neuraldata_session5_test]


# ## this has random element - do not rerun 

# s3_FRAMES_tr_sh, neuraldata_session3_tr_sh = aux_f4.shuffleNeuralData(neuraldata_session3, s3_indices_tr, 'A118')
# s3_FRAMES_test_sh, neuraldata_session3_test_sh = aux_f4.shuffleNeuralData(neuraldata_session3, s3_indices_test, 'A118')
# s4_FRAMES_tr_sh, neuraldata_session4_tr_sh = aux_f4.shuffleNeuralData(neuraldata_session4, s4_indices_tr, 'A118')
# s4_FRAMES_test_sh, neuraldata_session4_test_sh = aux_f4.shuffleNeuralData(neuraldata_session4, s4_indices_test, 'A118')
# s5_FRAMES_tr_sh, neuraldata_session5_tr_sh = aux_f4.shuffleNeuralData(neuraldata_session5, s5_indices_tr, 'A118')
# s5_FRAMES_test_sh, neuraldata_session5_test_sh = aux_f4.shuffleNeuralData(neuraldata_session5, s5_indices_test, 'A118')


# np.save('./information/indices_shuffled/A118_s3_indices_tr_sh.npy', s3_FRAMES_tr_sh)
# np.save('./information/indices_shuffled/A118_s3_indices_test_sh.npy', s3_FRAMES_test_sh)
# np.save('./information/indices_shuffled/A118_s4_indices_tr_sh.npy', s4_FRAMES_tr_sh)
# np.save('./information/indices_shuffled/A118_s4_indices_test_sh.npy', s4_FRAMES_test_sh)
# np.save('./information/indices_shuffled/A118_s5_indices_tr_sh.npy', s5_FRAMES_tr_sh)
# np.save('./information/indices_shuffled/A118_s5_indices_test_sh.npy', s5_FRAMES_test_sh)

# load these indices to use in this script 

s3_FRAMES_tr_sh = np.load('./information/indices_shuffled/A118_s3_indices_tr_sh.npy', allow_pickle=True)
s3_FRAMES_test_sh = np.load('./information/indices_shuffled/A118_s3_indices_test_sh.npy', allow_pickle=True)
s4_FRAMES_tr_sh = np.load('./information/indices_shuffled/A118_s4_indices_tr_sh.npy', allow_pickle=True)
s4_FRAMES_test_sh = np.load('./information/indices_shuffled/A118_s4_indices_test_sh.npy', allow_pickle=True)
s5_FRAMES_tr_sh = np.load('./information/indices_shuffled/A118_s5_indices_tr_sh.npy', allow_pickle=True)
s5_FRAMES_test_sh = np.load('./information/indices_shuffled/A118_s5_indices_test_sh.npy', allow_pickle=True)


# s3_filmID, s4_filmID, ... s3_frameID...s5_frameID already saved 

neuraldata_session3_tr_sh, s3_filmID_tr_sh, s3_frameID_tr_sh = sk.utils.shuffle(neuraldata_session3_tr, s3_filmID_tr, s3_frameID_tr, random_state=0)
neuraldata_session3_test_sh, s3_filmID_test_sh, s3_frameID_test_sh = sk.utils.shuffle(neuraldata_session3_test, s3_filmID_test, s3_frameID_test, random_state=1)
neuraldata_session4_tr_sh, s4_filmID_tr_sh, s4_frameID_tr_sh = sk.utils.shuffle(neuraldata_session4_tr, s4_filmID_tr, s4_frameID_tr, random_state=2)
neuraldata_session4_test_sh, s4_filmID_test_sh, s4_frameID_test_sh = sk.utils.shuffle(neuraldata_session4_test, s4_filmID_test, s4_frameID_test, random_state=3)
neuraldata_session5_tr_sh, s5_filmID_tr_sh, s5_frameID_tr_sh = sk.utils.shuffle(neuraldata_session5_tr, s5_filmID_tr, s5_frameID_tr, random_state=4)
neuraldata_session5_test_sh, s5_filmID_test_sh, s5_frameID_test_sh = sk.utils.shuffle(neuraldata_session5_test, s5_filmID_test, s5_frameID_test, random_state=5)


# save to information / neuraldata_shuffled

np.save('./information/neuraldata_shuffled/A118_s3_neuraldata_tr_sh.npy', neuraldata_session3_tr_sh)
np.save('./information/neuraldata_shuffled/A118_s3_neuraldata_test_sh.npy', neuraldata_session3_test_sh)
np.save('./information/neuraldata_shuffled/A118_s4_neuraldata_tr_sh.npy', neuraldata_session4_tr_sh)
np.save('./information/neuraldata_shuffled/A118_s4_neuraldata_test_sh.npy', neuraldata_session4_test_sh)
np.save('./information/neuraldata_shuffled/A118_s5_neuraldata_tr_sh.npy', neuraldata_session5_tr_sh)
np.save('./information/neuraldata_shuffled/A118_s5_neuraldata_test_sh.npy', neuraldata_session5_test_sh)

# load 

neuraldata_session3_tr_sh = np.load('./information/neuraldata_shuffled/A118_s3_neuraldata_tr_sh.npy', allow_pickle=True)
neuraldata_session3_test_sh = np.load('./information/neuraldata_shuffled/A118_s3_neuraldata_test_sh.npy', allow_pickle=True)
neuraldata_session4_tr_sh = np.load('./information/neuraldata_shuffled/A118_s4_neuraldata_tr_sh.npy', allow_pickle=True)
neuraldata_session4_test_sh = np.load('./information/neuraldata_shuffled/A118_s4_neuraldata_test_sh.npy', allow_pickle=True)
neuraldata_session5_tr_sh = np.load('./information/neuraldata_shuffled/A118_s5_neuraldata_tr_sh.npy', allow_pickle=True)
neuraldata_session5_test_sh = np.load('./information/neuraldata_shuffled/A118_s5_neuraldata_test_sh.npy', allow_pickle=True)

# save filmID's to information folder / filmID_shuffled 

np.save('./information/filmID_shuffled/A118_s3_filmID_tr_sh.npy', s3_filmID_tr_sh)
np.save('./information/filmID_shuffled/A118_s3_filmID_test_sh.npy', s3_filmID_test_sh)
np.save('./information/filmID_shuffled/A118_s4_filmID_tr_sh.npy', s4_filmID_tr_sh)
np.save('./information/filmID_shuffled/A118_s4_filmID_test_sh.npy', s4_filmID_test_sh)
np.save('./information/filmID_shuffled/A118_s5_filmID_tr_sh.npy', s5_filmID_tr_sh)
np.save('./information/filmID_shuffled/A118_s5_filmID_test_sh.npy', s5_filmID_test_sh)

# load 

s3_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s3_filmID_tr_sh.npy', allow_pickle=True)
s3_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s3_filmID_test_sh.npy', allow_pickle=True)
s4_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s4_filmID_tr_sh.npy', allow_pickle=True)
s4_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s4_filmID_test_sh.npy', allow_pickle=True)
s5_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s5_filmID_tr_sh.npy', allow_pickle=True)
s5_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s5_filmID_test_sh.npy', allow_pickle=True)


# # save FrameID to information folder / frameID_shuffled

np.save('./information/frameID_shuffled/A118_s3_frameID_tr_sh.npy', s3_frameID_tr_sh)
np.save('./information/frameID_shuffled/A118_s3_frameID_test_sh.npy', s3_frameID_test_sh)
np.save('./information/frameID_shuffled/A118_s4_frameID_tr_sh.npy', s4_frameID_tr_sh)
np.save('./information/frameID_shuffled/A118_s4_frameID_test_sh.npy', s4_frameID_test_sh)
np.save('./information/frameID_shuffled/A118_s5_frameID_tr_sh.npy', s5_frameID_tr_sh)
np.save('./information/frameID_shuffled/A118_s5_frameID_test_sh.npy', s5_frameID_test_sh)

# load 

s3_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s3_frameID_tr_sh.npy')
s3_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s3_frameID_test_sh.npy')
s4_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s4_frameID_tr_sh.npy')
s4_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s4_frameID_test_sh.npy')
s5_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s5_frameID_tr_sh.npy')
s5_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s5_frameID_test_sh.npy')

## ---- STATIONARY ------ ##

neuraldata_session3_tr_sh_st, s3_filmID_tr_st_sh, s3_frameID_tr_st_sh = sk.utils.shuffle(neuraldata_session3_tr_st, s3_filmID_tr_st, s3_frameID_tr_st, random_state=6)
neuraldata_session3_test_sh_st, s3_filmID_test_st_sh, s3_frameID_test_st_sh = sk.utils.shuffle(neuraldata_session3_test_st, s3_filmID_test_st, s3_frameID_test_st, random_state=7)
neuraldata_session4_tr_sh_st, s4_filmID_tr_st_sh, s4_frameID_tr_st_sh = sk.utils.shuffle(neuraldata_session4_tr_st, s4_filmID_tr_st, s4_frameID_tr_st, random_state=8)
neuraldata_session4_test_sh_st, s4_filmID_test_st_sh, s4_frameID_test_st_sh = sk.utils.shuffle(neuraldata_session4_test_st, s4_filmID_test_st, s4_frameID_test_st, random_state=9)
neuraldata_session5_tr_sh_st, s5_filmID_tr_st_sh, s5_frameID_tr_st_sh = sk.utils.shuffle(neuraldata_session5_tr_st, s5_filmID_tr_st, s5_frameID_tr_st, random_state=10)
neuraldata_session5_test_sh_st, s5_filmID_test_st_sh, s5_frameID_test_st_sh = sk.utils.shuffle(neuraldata_session5_test_st, s5_filmID_test_st, s5_frameID_test_st, random_state=11)

#save the neuraldata 

np.save('./information/neuraldata_st_shuffled/A118_s3_neuraldata_tr_st_sh.npy', neuraldata_session3_tr_sh_st)
np.save('./information/neuraldata_st_shuffled/A118_s3_neuraldata_test_st_sh.npy', neuraldata_session3_test_sh_st)
np.save('./information/neuraldata_st_shuffled/A118_s4_neuraldata_tr_st_sh.npy', neuraldata_session4_tr_sh_st)
np.save('./information/neuraldata_st_shuffled/A118_s4_neuraldata_test_st_sh.npy', neuraldata_session4_test_sh_st)
np.save('./information/neuraldata_st_shuffled/A118_s5_neuraldata_tr_st_sh.npy', neuraldata_session5_tr_sh_st)
np.save('./information/neuraldata_st_shuffled/A118_s5_neuraldata_test_st_sh.npy', neuraldata_session5_test_sh_st)

# load

neuraldata_session3_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s3_neuraldata_tr_st_sh.npy', allow_pickle=True)
neuraldata_session3_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s3_neuraldata_test_st_sh.npy', allow_pickle=True)
neuraldata_session4_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s4_neuraldata_tr_st_sh.npy', allow_pickle=True)
neuraldata_session4_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s4_neuraldata_test_st_sh.npy', allow_pickle=True)
neuraldata_session5_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s5_neuraldata_tr_st_sh.npy', allow_pickle=True)
neuraldata_session5_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s5_neuraldata_test_st_sh.npy', allow_pickle=True)


# # save filmID's to information folder / filmID_st_shuffled 

np.save('./information/filmID_st_shuffled/A118_s3_filmID_tr_st_sh.npy', s3_filmID_tr_st_sh)
np.save('./information/filmID_st_shuffled/A118_s3_filmID_test_st_sh.npy', s3_filmID_test_st_sh)
np.save('./information/filmID_st_shuffled/A118_s4_filmID_tr_st_sh.npy', s4_filmID_tr_st_sh)
np.save('./information/filmID_st_shuffled/A118_s4_filmID_test_st_sh.npy', s4_filmID_test_st_sh)
np.save('./information/filmID_st_shuffled/A118_s5_filmID_tr_st_sh.npy', s5_filmID_tr_st_sh)
np.save('./information/filmID_st_shuffled/A118_s5_filmID_test_st_sh.npy', s5_filmID_test_st_sh)

# load 

s3_filmID_tr_st_sh = np.load('./information/filmID_st_shuffled/A118_s3_filmID_tr_st_sh.npy', allow_pickle=True)
s3_filmID_test_st_sh = np.load('./information/filmID_st_shuffled/A118_s3_filmID_test_st_sh.npy', allow_pickle=True)
s4_filmID_tr_st_sh = np.load('./information/filmID_st_shuffled/A118_s4_filmID_tr_st_sh.npy', allow_pickle=True)
s4_filmID_test_st_sh = np.load('./information/filmID_st_shuffled/A118_s4_filmID_test_st_sh.npy', allow_pickle=True)
s5_filmID_tr_st_sh = np.load('./information/filmID_st_shuffled/A118_s5_filmID_tr_st_sh.npy', allow_pickle=True)
s5_filmID_test_st_sh = np.load('./information/filmID_st_shuffled/A118_s5_filmID_test_st_sh.npy', allow_pickle=True)

# # save FrameID to information folder / frameID_st_shuffled  

np.save('./information/frameID_st_shuffled/A118_s3_frameID_tr_st_sh.npy', s3_frameID_tr_st_sh)
np.save('./information/frameID_st_shuffled/A118_s3_frameID_test_st_sh.npy', s3_frameID_test_st_sh)
np.save('./information/frameID_st_shuffled/A118_s4_frameID_tr_st_sh.npy', s4_frameID_tr_st_sh)
np.save('./information/frameID_st_shuffled/A118_s4_frameID_test_st_sh.npy', s4_frameID_test_st_sh)
np.save('./information/frameID_st_shuffled/A118_s5_frameID_tr_st_sh.npy', s5_frameID_tr_st_sh)
np.save('./information/frameID_st_shuffled/A118_s5_frameID_test_st_sh.npy', s5_frameID_test_st_sh)

# load 

s3_frameID_tr_st_sh = np.load('./information/frameID_st_shuffled/A118_s3_frameID_tr_st_sh.npy', allow_pickle=True)
s3_frameID_test_st_sh = np.load('./information/frameID_st_shuffled/A118_s3_frameID_test_st_sh.npy', allow_pickle=True)
s4_frameID_tr_st_sh = np.load('./information/frameID_st_shuffled/A118_s4_frameID_tr_st_sh.npy', allow_pickle=True)
s4_frameID_test_st_sh = np.load('./information/frameID_st_shuffled/A118_s4_frameID_test_st_sh.npy', allow_pickle=True)
s5_frameID_tr_st_sh = np.load('./information/frameID_st_shuffled/A118_s5_frameID_tr_st_sh.npy', allow_pickle=True)
s5_frameID_test_st_sh = np.load('./information/frameID_st_shuffled/A118_s5_frameID_test_st_sh.npy', allow_pickle=True)

































# # get FilmID for these indices 

# # s3_filmID_sh = aux_f4.compilationSorterFilm(path_session3,chosenAnimal, neuraldata_session3.shape[2], 3)
# # s4_filmID_sh = aux_f4.compilationSorterFilm(path_session4,chosenAnimal, neuraldata_session4.shape[2], 4)
# # s5_filmID_sh = aux_f4.compilationSorterFilm(path_session5,chosenAnimal, neuraldata_session5.shape[2], 5)

# # s3_filmID_tr_sh = s3_filmID_sh[:, s3_indices_tr]
# # s3_filmID_test_sh = s3_filmID_sh[:, s3_indices_test]

# # s4_filmID_tr_sh = s4_filmID_sh[:, s4_indices_tr]
# # s4_filmID_test_sh = s4_filmID_sh[:, s4_indices_test]

# # s5_filmID_tr_sh = s5_filmID_sh[:, s5_indices_tr]
# # s5_filmID_test_sh = s5_filmID_sh[:, s5_indices_test]

# # # concatenate FilmID into a 1D array

# # s3_filmID_tr_sh = s3_filmID_tr_sh.flatten('F')
# # s3_filmID_test_sh = s3_filmID_test_sh.flatten('F')
# # s4_filmID_tr_sh = s4_filmID_tr_sh.flatten('F')
# # s4_filmID_test_sh = s4_filmID_test_sh.flatten('F')
# # s5_filmID_tr_sh = s5_filmID_tr_sh.flatten('F')
# # s5_filmID_test_sh = s5_filmID_test_sh.flatten('F')

# # shuffle these using aux_f4.shuffleData_1D()

# s3_filmID_tr_sh = aux_f4.shuffleData_1D(s3_filmID_tr, s3_FRAMES_tr_sh)
# s3_filmID_test_sh = aux_f4.shuffleData_1D(s3_filmID_test, s3_FRAMES_test_sh)
# s4_filmID_tr_sh = aux_f4.shuffleData_1D(s4_filmID_tr, s4_FRAMES_tr_sh)
# s4_filmID_test_sh = aux_f4.shuffleData_1D(s4_filmID_test, s4_FRAMES_test_sh)
# s5_filmID_tr_sh = aux_f4.shuffleData_1D(s5_filmID_tr, s5_FRAMES_tr_sh)
# s5_filmID_test_sh = aux_f4.shuffleData_1D(s5_filmID_test, s5_FRAMES_test_sh)

# # # get FrameID for these indices 

# # s3_frameID_sh = aux_f4.compilationSorterFrames(path_session3,chosenAnimal)
# # s4_frameID_sh = aux_f4.compilationSorterFrames(path_session4,chosenAnimal)
# # s5_frameID_sh = aux_f4.compilationSorterFrames(path_session5,chosenAnimal)

# # s3_frameID_tr_sh = s3_frameID_sh[:, s3_indices_tr]
# # s3_frameID_test_sh = s3_frameID_sh[:, s3_indices_test]

# # s4_frameID_tr_sh= s4_frameID_sh[:, s4_indices_tr]
# # s4_frameID_test_sh = s4_frameID_sh[:, s4_indices_test]

# # s5_frameID_tr_sh = s5_frameID_sh[:, s5_indices_tr]
# # s5_frameID_test_sh = s5_frameID_sh[:, s5_indices_test]

# # # concatenate FrameID into a 1D array

# # s3_frameID_tr_sh = s3_frameID_tr_sh.flatten('F')
# # s3_frameID_test_sh = s3_frameID_test_sh.flatten('F')
# # s4_frameID_tr_sh = s4_frameID_tr_sh.flatten('F')
# # s4_frameID_test_sh = s4_frameID_test_sh.flatten('F')
# # s5_frameID_tr_sh = s5_frameID_tr_sh.flatten('F')
# # s5_frameID_test_sh = s5_frameID_test_sh.flatten('F')

# # shuffle this 1D array using aux_f4.shuffleData_1D()

# s3_frameID_tr_sh = aux_f4.shuffleData_1D(s3_frameID_tr, s3_FRAMES_tr_sh)
# s3_frameID_test_sh = aux_f4.shuffleData_1D(s3_frameID_test, s3_FRAMES_test_sh)
# s4_frameID_tr_sh = aux_f4.shuffleData_1D(s4_frameID_tr, s4_FRAMES_tr_sh)
# s4_frameID_test_sh = aux_f4.shuffleData_1D(s4_frameID_test, s4_FRAMES_test_sh)
# s5_frameID_tr_sh = aux_f4.shuffleData_1D(s5_frameID_tr, s5_FRAMES_tr_sh)
# s5_frameID_test_sh = aux_f4.shuffleData_1D(s5_frameID_test, s5_FRAMES_test_sh)


# # # save filmID's to information folder / filmID_shuffled 

# # np.save('./information/filmID_shuffled/A118_s3_filmID_tr_sh.npy', s3_filmID_tr_sh)
# # np.save('./information/filmID_shuffled/A118_s3_filmID_test_sh.npy', s3_filmID_test_sh)
# # np.save('./information/filmID_shuffled/A118_s4_filmID_tr_sh.npy', s4_filmID_tr_sh)
# # np.save('./information/filmID_shuffled/A118_s4_filmID_test_sh.npy', s4_filmID_test_sh)
# # np.save('./information/filmID_shuffled/A118_s5_filmID_tr_sh.npy', s5_filmID_tr_sh)
# # np.save('./information/filmID_shuffled/A118_s5_filmID_test_sh.npy', s5_filmID_test_sh)

# # # load 

# # s3_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s3_filmID_tr_sh.npy', allow_pickle=True)
# # s3_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s3_filmID_test_sh.npy', allow_pickle=True)
# # s4_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s4_filmID_tr_sh.npy', allow_pickle=True)
# # s4_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s4_filmID_test_sh.npy', allow_pickle=True)
# # s5_filmID_tr_sh = np.load('./information/filmID_shuffled/A118_s5_filmID_tr_sh.npy', allow_pickle=True)
# # s5_filmID_test_sh = np.load('./information/filmID_shuffled/A118_s5_filmID_test_sh.npy', allow_pickle=True)



# # # # save FrameID to information folder / frameID_shuffled

# # np.save('./information/frameID_shuffled/A118_s3_frameID_tr_sh.npy', s3_frameID_tr_sh)
# # np.save('./information/frameID_shuffled/A118_s3_frameID_test_sh.npy', s3_frameID_test_sh)
# # np.save('./information/frameID_shuffled/A118_s4_frameID_tr_sh.npy', s4_frameID_tr_sh)
# # np.save('./information/frameID_shuffled/A118_s4_frameID_test_sh.npy', s4_frameID_test_sh)
# # np.save('./information/frameID_shuffled/A118_s5_frameID_tr_sh.npy', s5_frameID_tr_sh)
# # np.save('./information/frameID_shuffled/A118_s5_frameID_test_sh.npy', s5_frameID_test_sh)

# # # load 

# # s3_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s3_frameID_tr_sh.npy')
# # s3_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s3_frameID_test_sh.npy')
# # s4_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s4_frameID_tr_sh.npy')
# # s4_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s4_frameID_test_sh.npy')
# # s5_frameID_tr_sh = np.load('./information/frameID_shuffled/A118_s5_frameID_tr_sh.npy')
# # s5_frameID_test_sh = np.load('./information/frameID_shuffled/A118_s5_frameID_test_sh.npy')



# ## ---- STATIONARY ------ ##



# s3_FRAMES_tr_sh_st, neuraldata_session3_tr_sh_st = aux_f4.shuffleNeuralData(neuraldata_session3, s3_indices_tr_st, 'A118')
# s3_FRAMES_test_sh_st, neuraldata_session3_test_sh_st = aux_f4.shuffleNeuralData(neuraldata_session3, s3_indices_test_st, 'A118')
# s4_FRAMES_tr_sh_st, neuraldata_session4_tr_sh_st = aux_f4.shuffleNeuralData(neuraldata_session4, s4_indices_tr_st, 'A118')
# s4_FRAMES_test_sh_st, neuraldata_session4_test_sh_st = aux_f4.shuffleNeuralData(neuraldata_session4, s4_indices_test_st, 'A118')
# s5_FRAMES_tr_sh_st, neuraldata_session5_tr_sh_st = aux_f4.shuffleNeuralData(neuraldata_session5, s5_indices_tr_st, 'A118')
# s5_FRAMES_test_sh_st, neuraldata_session5_test_sh_st = aux_f4.shuffleNeuralData(neuraldata_session5, s5_indices_test_st, 'A118')

# # # save these generated indices in information / indices_st_shuffled

# # np.save('./information/indices_st_shuffled/A118_s3_indices_tr_st_sh.npy', s3_FRAMES_tr_sh_st)
# # np.save('./information/indices_st_shuffled/A118_s3_indices_test_st_sh.npy', s3_FRAMES_test_sh_st)
# # np.save('./information/indices_st_shuffled/A118_s4_indices_tr_st_sh.npy', s4_FRAMES_tr_sh_st)
# # np.save('./information/indices_st_shuffled/A118_s4_indices_test_st_sh.npy', s4_FRAMES_test_sh_st)
# # np.save('./information/indices_st_shuffled/A118_s5_indices_tr_st_sh.npy', s5_FRAMES_tr_sh_st)
# # np.save('./information/indices_st_shuffled/A118_s5_indices_test_st_sh.npy', s5_FRAMES_test_sh_st)

# # load 

# s3_FRAMES_tr_sh_st = np.load('./information/indices_st_shuffled/A118_s3_indices_tr_st_sh.npy', allow_pickle=True)
# s3_FRAMES_test_sh_st = np.load('./information/indices_st_shuffled/A118_s3_indices_test_st_sh.npy', allow_pickle=True)
# s4_FRAMES_tr_sh_st = np.load('./information/indices_st_shuffled/A118_s4_indices_tr_st_sh.npy', allow_pickle=True)
# s4_FRAMES_test_sh_st = np.load('./information/indices_st_shuffled/A118_s4_indices_test_st_sh.npy', allow_pickle=True)
# s5_FRAMES_tr_sh_st = np.load('./information/indices_st_shuffled/A118_s5_indices_tr_st_sh.npy', allow_pickle=True)
# s5_FRAMES_test_sh_st = np.load('./information/indices_st_shuffled/A118_s5_indices_test_st_sh.npy', allow_pickle=True)


# # # #save the neuraldata 

# # # np.save('./information/neuraldata_st_shuffled/A118_s3_neuraldata_tr_st_sh.npy', neuraldata_session3_tr_sh_st)
# # # np.save('./information/neuraldata_st_shuffled/A118_s3_neuraldata_test_st_sh.npy', neuraldata_session3_test_sh_st)
# # # np.save('./information/neuraldata_st_shuffled/A118_s4_neuraldata_tr_st_sh.npy', neuraldata_session4_tr_sh_st)
# # # np.save('./information/neuraldata_st_shuffled/A118_s4_neuraldata_test_st_sh.npy', neuraldata_session4_test_sh_st)
# # # np.save('./information/neuraldata_st_shuffled/A118_s5_neuraldata_tr_st_sh.npy', neuraldata_session5_tr_sh_st)
# # # np.save('./information/neuraldata_st_shuffled/A118_s5_neuraldata_test_st_sh.npy', neuraldata_session5_test_sh_st)

# # # load

# # neuraldata_session3_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s3_neuraldata_tr_st_sh.npy', allow_pickle=True)
# # neuraldata_session3_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s3_neuraldata_test_st_sh.npy', allow_pickle=True)
# # neuraldata_session4_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s4_neuraldata_tr_st_sh.npy', allow_pickle=True)
# # neuraldata_session4_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s4_neuraldata_test_st_sh.npy', allow_pickle=True)
# # neuraldata_session5_tr_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s5_neuraldata_tr_st_sh.npy', allow_pickle=True)
# # neuraldata_session5_test_sh_st = np.load('./information/neuraldata_st_shuffled/A118_s5_neuraldata_test_st_sh.npy', allow_pickle=True)


# # # get FilmID for these indices 

# # s3_filmID_tr_st_sh = s3_filmID[:, s3_indices_tr_st]
# # s3_filmID_test_st_sh = s3_filmID[:, s3_indices_test_st]

# # s4_filmID_tr_st_sh = s4_filmID[:, s4_indices_tr_st]
# # s4_filmID_test_st_sh = s4_filmID[:, s4_indices_test_st]

# # s5_filmID_tr_st_sh = s5_filmID[:, s5_indices_tr_st]
# # s5_filmID_test_st_sh = s5_filmID[:, s5_indices_test_st]

# # # concatenate FilmID into a 1D array

# # s3_filmID_tr_st_sh = s3_filmID_tr_st_sh.flatten('F')
# # s3_filmID_test_st_sh = s3_filmID_test_st_sh.flatten('F')
# # s4_filmID_tr_st_sh = s4_filmID_tr_st_sh.flatten('F')
# # s4_filmID_test_st_sh = s4_filmID_test_st_sh.flatten('F')
# # s5_filmID_tr_st_sh = s5_filmID_tr_st_sh.flatten('F')
# # s5_filmID_test_st_sh = s5_filmID_test_st_sh.flatten('F')

# # shuffle 

# s3_filmID_tr_st_sh = aux_f4.shuffleData_1D(s3_filmID_tr_st, s3_FRAMES_tr_sh_st)
# s3_filmID_test_st_sh = aux_f4.shuffleData_1D(s3_filmID_test_st, s3_FRAMES_test_sh_st)
# s4_filmID_tr_st_sh = aux_f4.shuffleData_1D(s4_filmID_tr_st, s4_FRAMES_tr_sh_st)
# s4_filmID_test_st_sh = aux_f4.shuffleData_1D(s4_filmID_test_st, s4_FRAMES_test_sh_st)
# s5_filmID_tr_st_sh = aux_f4.shuffleData_1D(s5_filmID_tr_st, s5_FRAMES_tr_sh_st)
# s5_filmID_test_st_sh = aux_f4.shuffleData_1D(s5_filmID_test_st, s5_FRAMES_test_sh_st)


# # # get FrameID for these indices 

# # s3_frameID_st_sh = aux_f4.compilationSorterFrames(path_session3,chosenAnimal)
# # s4_frameID_st_sh = aux_f4.compilationSorterFrames(path_session4,chosenAnimal)
# # s5_frameID_st_sh = aux_f4.compilationSorterFrames(path_session5,chosenAnimal)

# # s3_frameID_tr_st_sh = s3_frameID[:, s3_indices_tr_st]
# # s3_frameID_test_st_sh = s3_frameID[:, s3_indices_test_st]

# # s4_frameID_tr_st_sh= s4_frameID[:, s4_indices_tr_st]
# # s4_frameID_test_st_sh = s4_frameID[:, s4_indices_test_st]

# # s5_frameID_tr_st_sh = s5_frameID[:, s5_indices_tr_st]
# # s5_frameID_test_st_sh = s5_frameID[:, s5_indices_test_st]

# # # concatenate FrameID into a 1D array

# # s3_frameID_tr_st_sh = s3_frameID_tr_st_sh.flatten('F')
# # s3_frameID_test_st_sh = s3_frameID_test_st_sh.flatten('F')
# # s4_frameID_tr_st_sh = s4_frameID_tr_st_sh.flatten('F')
# # s4_frameID_test_st_sh = s4_frameID_test_st_sh.flatten('F')
# # s5_frameID_tr_st_sh = s5_frameID_tr_st_sh.flatten('F')
# # s5_frameID_test_st_sh = s5_frameID_test_st_sh.flatten('F')

# # # shuffle 

# s3_frameID_tr_st_sh = aux_f4.shuffleData_1D(s3_frameID_tr_st, s3_FRAMES_tr_sh_st)
# s3_frameID_test_st_sh = aux_f4.shuffleData_1D(s3_frameID_test_st, s3_FRAMES_test_sh_st)
# s4_frameID_tr_st_sh = aux_f4.shuffleData_1D(s4_frameID_tr_st, s4_FRAMES_tr_sh_st)
# s4_frameID_test_st_sh = aux_f4.shuffleData_1D(s4_frameID_test_st, s4_FRAMES_test_sh_st)
# s5_frameID_tr_st_sh = aux_f4.shuffleData_1D(s5_frameID_tr_st, s5_FRAMES_tr_sh_st)
# s5_frameID_test_st_sh = aux_f4.shuffleData_1D(s5_frameID_test_st, s5_FRAMES_test_sh_st)

# # # # save filmID's to information folder / filmID_st_shuffled 

# # np.save('./information/filmID_st_shuffled/A118_s3_filmID_tr_st_sh.npy', s3_filmID_tr_st_sh)
# # np.save('./information/filmID_st_shuffled/A118_s3_filmID_test_st_sh.npy', s3_filmID_test_st_sh)
# # np.save('./information/filmID_st_shuffled/A118_s4_filmID_tr_st_sh.npy', s4_filmID_tr_st_sh)
# # np.save('./information/filmID_st_shuffled/A118_s4_filmID_test_st_sh.npy', s4_filmID_test_st_sh)
# # np.save('./information/filmID_st_shuffled/A118_s5_filmID_tr_st_sh.npy', s5_filmID_tr_st_sh)
# # np.save('./information/filmID_st_shuffled/A118_s5_filmID_test_st_sh.npy', s5_filmID_test_st_sh)

# # # load 

# # s3_filmID_tr_st_sh = np.load('./information/filmID_st_shuffled/A118_s3_filmID_tr_st_sh.npy', allow_pickle=True)
# # s3_filmID_test_st_sh = np.load('./information/filmID_st_shuffled/A118_s3_filmID_test_st_sh.npy', allow_pickle=True)
# # s4_filmID_tr_st_sh = np.load('./information/filmID_st_shuffled/A118_s4_filmID_tr_st_sh.npy', allow_pickle=True)
# # s4_filmID_test_st_sh = np.load('./information/filmID_st_shuffled/A118_s4_filmID_test_st_sh.npy', allow_pickle=True)
# # s5_filmID_tr_st_sh = np.load('./information/filmID_st_shuffled/A118_s5_filmID_tr_st_sh.npy', allow_pickle=True)
# # s5_filmID_test_st_sh = np.load('./information/filmID_st_shuffled/A118_s5_filmID_test_st_sh.npy', allow_pickle=True)

# # # # save FrameID to information folder / frameID_st_shuffled  

# # np.save('./information/frameID_st_shuffled/A118_s3_frameID_tr_st_sh.npy', s3_frameID_tr_st_sh)
# # np.save('./information/frameID_st_shuffled/A118_s3_frameID_test_st_sh.npy', s3_frameID_test_st_sh)
# # np.save('./information/frameID_st_shuffled/A118_s4_frameID_tr_st_sh.npy', s4_frameID_tr_st_sh)
# # np.save('./information/frameID_st_shuffled/A118_s4_frameID_test_st_sh.npy', s4_frameID_test_st_sh)
# # np.save('./information/frameID_st_shuffled/A118_s5_frameID_tr_st_sh.npy', s5_frameID_tr_st_sh)
# # np.save('./information/frameID_st_shuffled/A118_s5_frameID_test_st_sh.npy', s5_frameID_test_st_sh)

# # # load 

# # s3_frameID_tr_st_sh = np.load('./information/frameID_st_shuffled/A118_s3_frameID_tr_st_sh.npy', allow_pickle=True)
# # s3_frameID_test_st_sh = np.load('./information/frameID_st_shuffled/A118_s3_frameID_test_st_sh.npy', allow_pickle=True)
# # s4_frameID_tr_st_sh = np.load('./information/frameID_st_shuffled/A118_s4_frameID_tr_st_sh.npy', allow_pickle=True)
# # s4_frameID_test_st_sh = np.load('./information/frameID_st_shuffled/A118_s4_frameID_test_st_sh.npy', allow_pickle=True)
# # s5_frameID_tr_st_sh = np.load('./information/frameID_st_shuffled/A118_s5_frameID_tr_st_sh.npy', allow_pickle=True)
# # s5_frameID_test_st_sh = np.load('./information/frameID_st_shuffled/A118_s5_frameID_test_st_sh.npy', allow_pickle=True)





