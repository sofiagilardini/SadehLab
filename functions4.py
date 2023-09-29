import numpy as np 
import matplotlib.pyplot as plt
import os
import fnmatch
import classes 
import random
import cebra
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def getneuraldata(animal : str, experiment : str, colour: str, stationaryfilter: bool):

    # experiment: 'Crickets', 'Compilation', ?
    # given an animal and an experiment, return list of paths
    # colour : red = visual , green = auditory
    # stationary = True (applies the stationary filter)

    print('got here')
    #directory = '/Users/sgb/Documents/git/CEBRA/Sofia/full_dataset_CRICK/' + str(animal)
    directory = os.getcwd() + '/' + str(animal)
    print("here: directory", directory)

    list_paths = []

    # Get the sorted list of directory entries
    # entries = sorted(os.scandir(directory), key=lambda entry: int(entry.name[1:])) ?????
    entries = os.scandir(directory)

    print(entries)

    # iterate over entries in the directory
    for entry in entries:

        if entry.is_dir():
            subdirectory = entry.path
            # keep this to check order is 2, 3, 4, 5, 6
            # check if Razer computes this the same way / why does sorted not work? 
            print('entry path: ', entry.path)


            for filename in os.listdir(subdirectory):
                f = os.path.join(subdirectory, filename)

                if os.path.isfile(f):

                    if f.__contains__(str(experiment)) and f.__contains__('F') and f.__contains__(str(colour)):
                        list_paths.append(f)

    return(list_paths)

def getBehaviourPath(path_test: str):

    f = path_test

    neuraldata = np.load(f, allow_pickle=True)
    behaviourpath = str()

    for i in range(len(path_test)):
        if path_test[i] != "F" :
            behaviourpath += str(path_test[i])
        
        else:
            break

    behaviourpath = behaviourpath + "behaviourData_clean.npy"

    return behaviourpath

def getStationaryIndex(path_test: str):

    # this function is only for sessions 3-5. where trialID is the same across all trials

    behaviourpath = getBehaviourPath(path_test)

    behaviourdata = np.load(behaviourpath, allow_pickle=True)
    encoderData = behaviourdata.item()['encoder']
    encoder_mean = np.mean(encoderData, axis = 1)
    indices_stationary = np.where(encoder_mean <= 0)
    index1, index2, index3 = indices_stationary

    return index2

def getBehaviourData(path_test: str, behaviourtype: str):
    # this function is only for sessions 3-5. where trialID is the same across all trials

    behaviourpath = getBehaviourPath(path_test)

    behaviourdata = np.load(behaviourpath, allow_pickle=True)

    if behaviourtype == "encoder":
        BehaviourData = behaviourdata.item()['encoder']

    if behaviourtype == "pupilDiam":
        BehaviourData = behaviourdata.item()['pupilDiam']

    return BehaviourData

# to delete 
# def concatenateBehaviourData(behaviourData, indices, animal: str):

#     if animal == "A118":
#         A118 = classes.A118

#     i = 0
#     # create concatenated data for training

#     for trial in indices:
#         behaviourData_ = behaviourData[:,:,trial, 0]
#         behaviourData_ = behaviourData_[:, A118.startframe:A118.endframe].T

#         if i == 0:
#             concat_data = behaviourData_
        
#         else:
#             concat_data = np.concatenate((concat_data, behaviourData_)) 

#         i += 1
    
#     return behaviourData_

def concatenate_neuraldata(neuraldata, indices, animal: str):

    if animal == "A118":
        A118 = classes.A118

    i = 0
    # create concatenated data for training

    for trial in indices:
        neuraldata_tr = neuraldata[:,:,trial, 0]
        neuraldata_tr = neuraldata_tr[:, A118.startframe:A118.endframe].T

        if i == 0:
            concat_neuraldata = neuraldata_tr
        
        else:
            concat_neuraldata = np.concatenate((concat_neuraldata, neuraldata_tr)) 

        i += 1
    
    return concat_neuraldata

def concatenateData(neuraldata, indices, animal: str):

    if animal == "A118":
        A118 = classes.A118

    i = 0
    # create concatenated data for training

    for trial in indices:
        neuraldata_tr = neuraldata[:,:,trial, 0]
        neuraldata_tr = neuraldata_tr[:, A118.startframe:A118.endframe].T

        if i == 0:
            concat_neuraldata = neuraldata_tr
        
        else:
            concat_neuraldata = np.concatenate((concat_neuraldata, neuraldata_tr)) 

        i += 1
    
    return concat_neuraldata

def compilationSorterFilm(path_test: str, animal: str, indices: list, session: int):

    # check video is compilation using os module
    # if it is, return list of 0, 1, 2, 3 depending on which film is being watched

    # iterate over entries in the directory

    testanimal = str(animal)

    neuraldata = np.load(path_test, allow_pickle=True)

    if path_test.__contains__("Compilation"):

        if testanimal == "A118":
            print("animal is A118")
            testanimal = classes.A118

        else:
            print('ANIMAL NOT FOUND')
            return None
        
        total_frames = testanimal.endframe - testanimal.startframe
        total_time = total_frames * 1/testanimal.s_rate

        film_discrete = np.zeros((total_frames, neuraldata.shape[2]), dtype = int)
        
        for trial in range(neuraldata.shape[2]):
            for frame in range(testanimal.startframe, testanimal.endframe):

                frame = frame - testanimal.startframe
                
                #print('time in seconds', frame * 1/testanimal.s_rate)
                time_s = frame * 1/testanimal.s_rate

                if time_s < 2.5:
                    film_discrete[frame, trial] = 0
                    #print('0')
                if time_s >= 2.5 and time_s < 5.77:
                    film_discrete[frame, trial] = 1
                    #print('1')
                if time_s >= 5.77 and time_s < 10.05:
                    film_discrete[frame, trial] = 2
                    #print('2')  
                if time_s >= 10.05 and time_s < 12.13:
                    film_discrete[frame, trial] = 3
                    #print('3')
                if time_s >= 12.13:
                    film_discrete[frame, trial] = 4
                    print('error')

        #Example data: frames x trials array   
        print('shape film_discrete', film_discrete.shape)

        # Plot the tile-like plot using the "viridis" colormap
        # NB: plt.imshow plots counterintuitively, so we transpose the data


        plt.imshow(film_discrete.T, cmap='viridis', aspect='auto')

        # Set x and y-axis labels
        plt.xlabel('Frames')
        plt.ylabel('Trials')

        # Set color bar and its labels
        cbar = plt.colorbar(ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['0', '1', '2', '3'])

        plt.savefig('./information/figures/compilationSorterFilm' + str(session))

        #plt.show()
        # Show the plot
        
    

    else:
        return False

    return film_discrete

# def concatenate_behaviour(data, indices, animal: str):
    
#     if animal == "A118":
#         A118 = classes.A118

#     i = 0
    
#     for trial in indices:
#         data_temp = data[:,:,trial, 0]
#         data_temp = data_temp[:, A118.startframe:A118.endframe].T

#         if i == 0:
#             concat_data = data_temp
        
#         else:
#             concat_data = np.concatenate((concat_data, data_temp)) 

#         i += 1
    
#     return concat_data

def trainCebraBehaviour(model, neuraldata_tr, neuraldata_test, behaviourdata_tr, behaviourdata_test, session: int, modelID: str, ind_tr, ind_test, modeltype: str):

    model.fit(neuraldata_tr, behaviourdata_tr)
    
    model_name = "./training_results/" + str(modeltype) + "/cebra_model_" + str(modelID) + ".pt"

    # send model to cpu and save
    model.to('cpu')        
    model.save(model_name)

    text_file_name = "./training_results/" + str(modeltype) + "/training_info_" + str(modelID) + ".txt"

    with open(text_file_name, 'w') as text_file:
        text_file.write("session: " + str(session) + '\n')
        text_file.write("indices training: " + str(ind_tr) + '\n')
        text_file.write("indices testing: " + str(ind_test) + '\n')
        text_file.write("model architecture: " + str(model.model_architecture) + '\n')

    ax2 = cebra.plot_loss(model)
    fig_name = "./training_results/" + str(modeltype) + "/training_loss_" + str(modelID)
    
    plt.savefig(fig_name)

    #plt.show()


    return model



def trainCebraBehaviour_dual(model, neuraldata_tr, continuous_label_tr, discrete_label_tr, session: int, modelID: str, ind_tr, ind_test, modeltype: str):

    model.fit(neuraldata_tr, continuous_label_tr, discrete_label_tr)
    
    model_name = "./training_results/" + str(modeltype) + "/cebra_model_" + str(modelID) + ".pt"

    # send model to cpu and save
    model.to('cpu')        
    model.save(model_name)

    text_file_name = "./training_results/" + str(modeltype) + "/training_info_" + str(modelID) + ".txt"

    with open(text_file_name, 'w') as text_file:
        text_file.write("session: " + str(session) + '\n')
        text_file.write("indices training: " + str(ind_tr) + '\n')
        text_file.write("indices testing: " + str(ind_test) + '\n')
        text_file.write("model architecture: " + str(model.model_architecture) + '\n')


    ax2 = cebra.plot_loss(model)
    fig_name = "./training_results/" + str(modeltype) + "/training_loss_" + str(modelID)
    
    plt.savefig(fig_name)

    #plt.show()


    return model

def compilationSorterFrames(path_test: str, animal: str):

    # check video is compilation using os module
    # if it is, return list of 0, 1, 2, 3 depending on which film is being watched

    # iterate over entries in the directory

    testanimal = str(animal)

    neuraldata = np.load(path_test, allow_pickle=True)

    print('shape neuraldata', neuraldata.shape)

    if path_test.__contains__("Compilation"):

        if testanimal == "A118":
            print("animal is A118")
            testanimal = classes.A118

        else:
            print('ANIMAL NOT FOUND')
            return None
        

        #print('within func, shape neuraldata', neuraldata.shape)

        total_frames = testanimal.endframe - testanimal.startframe
        total_time = total_frames * 1/testanimal.s_rate
        #print('total time', total_time)
        #print('total frames', total_frames)

        compilation_frames = np.zeros((total_frames, neuraldata.shape[2]), dtype = int)
        
        for trial in range(neuraldata.shape[2]):

            framecounter = 0

            for frame in range(testanimal.startframe, testanimal.endframe):
                frame = frame - testanimal.startframe
                compilation_frames[frame, trial] = framecounter
                framecounter += 1

        plt.imshow(compilation_frames.T, cmap='viridis', aspect='auto')

        #Set x and y-axis labels
        plt.xlabel('Frames')
        plt.ylabel('Trials')

        #Set color bar and its labels

        #Show the plot
        #plt.show()

        return compilation_frames

def cKNN(model, neuraldata_tr, neuraldata_test, behaviour_train, behaviour_test):

    train_embedding = model.transform(neuraldata_tr)
    test_embedding = model.transform(neuraldata_test)

    cKNN_decoder = cebra.KNNDecoder(n_neighbors = 36, metric = 'cosine')

    cKNN_decoder.fit(train_embedding, behaviour_train)

    cKNN_pred = cKNN_decoder.predict(test_embedding)

    accuracy_cKNN = accuracy_score(behaviour_test, cKNN_pred)


    return accuracy_cKNN 


def plot_CKNN_KNN(knn_acc, cknn_acc, test_type: str):

    # input lists and then use those to plot
                  
    # plot accuracy of C-KNN 

    barWidth = 0.1
    fig = plt.subplots(figsize=(12, 8))

    S3_knn = [knn_acc[0], knn_acc[1], knn_acc[2]]
    S3_cknn = [cknn_acc[0], cknn_acc[1], cknn_acc[2]]

    S4_knn = [knn_acc[3], knn_acc[4], knn_acc[5]]
    S4_cknn = [cknn_acc[3], cknn_acc[4], cknn_acc[5]]

    S5_knn = [knn_acc[6], knn_acc[7], knn_acc[8]]
    S5_cknn = [cknn_acc[6], cknn_acc[7], cknn_acc[8]]


    # Set position of bar on X axis
    br1 = np.arange(len(S3_knn))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]

    # dark colours are cknn, light colours are knn

    if 'filmID' in test_type:

        color_s3_cknn = (0.48, 0.48, 0.48)  # Dark grey (slightly darker)
        color_s3_knn = (0.72, 0.72, 0.72, 0.4)  # Light grey (slightly lighter)

        color_s4_cknn = (0.18, 0.38, 0.58)  # Dark blue (slightly darker)
        color_s4_knn = (0.42, 0.62, 0.82, 0.4)  # Light blue (slightly lighter)

        color_s5_cknn = (0.48, 0.28, 0.48)  # Dark purple (slightly darker)
        color_s5_knn = (0.72, 0.52, 0.72, 0.4)  # Light purple (slightly lighter)

    
    if 'frameID' in test_type:

        # Updated colors with adjustments
        color_s3_cknn = (1.0, 0.7, 0.0)  # Slightly Orange Yellow (slightly darker)
        color_s3_knn = (1.0, 0.8, 0.3, 0.4)  # Pale Orange Yellow (slightly lighter)

        color_s4_cknn = (0.6, 0.2, 0.2)  # Maroon (slightly darker)
        color_s4_knn = (0.8, 0.4, 0.4, 0.4)  # Light Maroon (slightly lighter)

        color_s5_cknn = (0.0, 0.6, 0.7)  # Dark Turquoise (slightly darker)
        color_s5_knn = (0.4, 0.7, 0.8, 0.4)  # Light Turquoise (slightly lighter)



    # Add numeric values to bars with smaller font and 2 decimal places
    plt.bar_label(plt.bar(br1, S3_knn, color=color_s3_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S3 KNN'), label_type='center', fontsize=8, fmt='%.2f')
    plt.bar_label(plt.bar(br2, S3_cknn, color=color_s3_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S3 cKNN'), label_type='center', fontsize=8, fmt='%.2f')

    plt.bar_label(plt.bar(br3, S4_knn, color=color_s4_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S4 KNN'), label_type='center', fontsize=8, fmt='%.2f')
    plt.bar_label(plt.bar(br4, S4_cknn, color=color_s4_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S4 cKNN'), label_type='center', fontsize=8, fmt='%.2f')

    plt.bar_label(plt.bar(br5, S5_knn, color=color_s5_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S5 KNN'), label_type='center', fontsize=8, fmt='%.2f')
    plt.bar_label(plt.bar(br6, S5_cknn, color=color_s5_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S5 cKNN'), label_type='center', fontsize=8, fmt='%.2f')

    # Calculate the center positions of each group for x-tick placement
    br_centers = [(a + b) / 2.0 for a, b in zip(br1, br6)]
    # Adding Xticks
    plt.title('Accuracy of KNN vs CEBRA-KNN decoding across sessions ' + str(test_type), fontweight='bold', fontsize=10)
    # plt.xlabel('Model', fontsize=10)
    plt.ylabel('Accuracy', fontsize=13)
    plt.xticks(br_centers, ['model trained on S3', 'model trained on S4', 'model trained on S5'], fontsize = 13)
    plt.grid(axis='y', alpha=0.75, linestyle='-', linewidth=0.25)

    # Optimize legend placement (loc='upper left' may work well)
    plt.legend(loc='upper right', bbox_to_anchor=(1.12,1.11), fontsize = 9)
    plt.ylim(0,1)
    plt.savefig('./information/temporary/KNN_CKNN_' + str(test_type) + '.png')

    #plt.show()

def plot_CKNN_KNN_sh(knn_acc, cknn_acc, cknn_acc_sh, test_type: str):

    barWidth = 0.1
    fig, ax = plt.subplots(figsize=(12, 8))

    S3_knn = [knn_acc[0], knn_acc[1], knn_acc[2]]
    S3_cknn = [cknn_acc[0], cknn_acc[1], cknn_acc[2]]
    S3_cknn_sh = [cknn_acc_sh[0], cknn_acc_sh[1], cknn_acc_sh[2]]

    S4_knn = [knn_acc[3], knn_acc[4], knn_acc[5]]
    S4_cknn = [cknn_acc[3], cknn_acc[4], cknn_acc[5]]
    S4_cknn_sh = [cknn_acc_sh[3], cknn_acc_sh[4], cknn_acc_sh[5]]

    S5_knn = [knn_acc[6], knn_acc[7], knn_acc[8]]
    S5_cknn = [cknn_acc[6], cknn_acc[7], cknn_acc[8]]
    S5_cknn_sh = [cknn_acc_sh[6], cknn_acc_sh[7], cknn_acc_sh[8]]

    # Set position of bar on X axis
    br1 = np.arange(len(S3_knn))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]
    br7 = [x + barWidth for x in br6]
    br8 = [x + barWidth for x in br7]
    br9 = [x + barWidth for x in br8]

    # dark colours are cknn, light colours are knn

    if 'filmID' in test_type:

        color_s3_cknn = (0.48, 0.48, 0.48)  # Dark grey (slightly darker)
        color_s3_knn = (0.72, 0.72, 0.72, 0.8)  # Light grey (slightly lighter)
        color_s3_cknn_sh = (0.6, 0.48, 0.48, 0.4)

        color_s4_cknn = (0.18, 0.38, 0.58)  # Dark blue (slightly darker)
        color_s4_knn = (0.42, 0.62, 0.82, 0.8)  # Light blue (slightly lighter)
        color_s4_cknn_sh = (0.38, 0.38, 0.58, 0.4)


        color_s5_cknn = (0.48, 0.28, 0.48)  # Dark purple (slightly darker)
        color_s5_knn = (0.72, 0.52, 0.72, 0.8)  # Light purple (slightly lighter)
        color_s5_cknn_sh = (0.78, 0.28, 0.48, 0.4)

    
    if 'frameID' in test_type:

        # Updated colors with adjustments
        color_s3_cknn = (1.0, 0.7, 0.0)  # Slightly Orange Yellow (slightly darker)
        color_s3_knn = (1.0, 0.8, 0.3, 0.8)  # Pale Orange Yellow (slightly lighter)
        color_s3_cknn_sh = (1.0, 0.4, 0.0, 0.4)

        color_s4_cknn = (0.6, 0.2, 0.2)  # Maroon (slightly darker)
        color_s4_knn = (0.8, 0.4, 0.4, 0.8)  # Light Maroon (slightly lighter)
        color_s4_cknn_sh = (0.6, 0, 0.2, 0.4)  # Maroon (slightly darker)

        color_s5_cknn = (0.0, 0.6, 0.7)  # Dark Turquoise (slightly darker)
        color_s5_knn = (0.4, 0.7, 0.8, 0.8)  # Light Turquoise (slightly lighter)
        color_s5_cknn_sh = (0.0, 0.2, 0.7, 0.4)  # Dark Turquoise (slightly darker)

    # Plot horizontal lines for KNN
    for i in range(len(S3_knn)):
        ax.hlines(S3_knn[i], br1[i] - barWidth + 0.05, br1[i] + 1.5 * barWidth, colors=color_s3_knn, linewidth=2,)
        ax.hlines(S4_knn[i], br4[i] - barWidth + 0.05, br4[i] + 1.5 * barWidth, colors=color_s4_knn, linewidth=2,)
        ax.hlines(S5_knn[i], br7[i] - barWidth + 0.05, br7[i] + 1.5 * barWidth, colors=color_s5_knn, linewidth=2,)

    # Add numeric values to bars with smaller font and 2 decimal places
    #ax.bar_label(ax.bar(br1, S3_knn, color=color_s3_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S3 KNN'), label_type='center', fontsize=0.5, fmt='%.2f')

    #ax.bar_label(ax.bar(br1, S3_knn, color=color_s3_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S3 KNN'), label_type='center', fontsize=8)
    ax.bar_label(ax.bar(br1, S3_cknn, color=color_s3_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S3 cKNN'), label_type='center', fontsize=8, fmt='%.2f')
    ax.bar_label(ax.bar(br2, S3_cknn_sh, color=color_s3_cknn_sh, width=barWidth, edgecolor='grey', linewidth=0.25, label='S3 cKNN_SH'), label_type='center', fontsize=8, fmt='%.2f')

    #ax.bar_label(ax.bar(br4, S4_knn, color=color_s4_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S4 KNN'), label_type='center', fontsize=0.5, fmt='%.2f')
    #ax.bar_label(ax.bar(br4, S4_knn, color=color_s4_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S4 KNN'), label_type='center', fontsize=8)

    ax.bar_label(ax.bar(br4, S4_cknn, color=color_s4_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S4 cKNN'), label_type='center', fontsize=8, fmt='%.2f')
    ax.bar_label(ax.bar(br5, S4_cknn_sh, color=color_s4_cknn_sh, width=barWidth, edgecolor='grey', linewidth=0.25, label='S4 cKNN_SH'), label_type='center', fontsize=8, fmt='%.2f')

    #ax.bar_label(ax.bar(br7, S5_knn, color=color_s5_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S5 KNN'), label_type='center', fontsize=0.5, fmt='%.2f')
    #ax.bar_label(ax.bar(br7, S5_knn, color=color_s5_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S5 KNN'), label_type='center', fontsize=8)
    ax.bar_label(ax.bar(br7, S5_cknn, color=color_s5_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S5 cKNN'), label_type='center', fontsize=8, fmt='%.2f')
    ax.bar_label(ax.bar(br8, S5_cknn_sh, color=color_s5_cknn_sh, width=barWidth, edgecolor='grey', linewidth=0.25, label='S5 cKNN_SH'), label_type='center', fontsize=8, fmt='%.2f')

    # Calculate the center positions of each group for x-tick placement
    br_centers = [(a + b) / 2.0 for a, b in zip(br1, br9)]
    # Adding Xticks
    plt.title('Accuracy of KNN vs CEBRA-KNN decoding across sessions ' + str(test_type), fontweight='bold', fontsize=10)
    # plt.xlabel('Model', fontsize=10)
    plt.ylabel('Accuracy', fontsize=13)
    plt.xticks(br_centers, ['model trained on S3', 'model trained on S4', 'model trained on S5'], fontsize = 13)
    plt.grid(axis='y', alpha=0.75, linestyle='-', linewidth=0.25)

    # Optimize legend placement (loc='upper left' may work well)
    plt.legend(loc='upper right', bbox_to_anchor=(1.13,1.11), fontsize = 8)
    plt.ylim(0,1)
    plt.savefig('./information/temporary/KNN_CKNN_sh_delta' + str(test_type) + '.png')

    #plt.show()

def plotKNN_pupilDiam(knn_acc, cknn_acc, test_type: str):

    # input lists and then use those to plot
                  
    # plot accuracy of C-KNN 

    barWidth = 0.1
    fig = plt.subplots(figsize=(12, 8))

    S3_knn = [knn_acc[0], knn_acc[1], knn_acc[2]]
    S3_cknn = [cknn_acc[0], cknn_acc[1], cknn_acc[2]]

    S4_knn = [knn_acc[3], knn_acc[4], knn_acc[5]]
    S4_cknn = [cknn_acc[3], cknn_acc[4], cknn_acc[5]]

    S5_knn = [knn_acc[6], knn_acc[7], knn_acc[8]]
    S5_cknn = [cknn_acc[6], cknn_acc[7], cknn_acc[8]]


    # Set position of bar on X axis
    br1 = np.arange(len(S3_knn))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]

    # dark colours are cknn, light colours are knn

    if 'filmID' in test_type:

        color_s3_cknn = (0.48, 0.48, 0.48)  # Dark grey (slightly darker)
        color_s3_knn = (0.72, 0.72, 0.72, 0.4)  # Light grey (slightly lighter)

        color_s4_cknn = (0.18, 0.38, 0.58)  # Dark blue (slightly darker)
        color_s4_knn = (0.42, 0.62, 0.82, 0.4)  # Light blue (slightly lighter)

        color_s5_cknn = (0.48, 0.28, 0.48)  # Dark purple (slightly darker)
        color_s5_knn = (0.72, 0.52, 0.72, 0.4)  # Light purple (slightly lighter)

    
    if 'frameID' in test_type:

        # Updated colors with adjustments
        color_s3_cknn = (1.0, 0.7, 0.0)  # Slightly Orange Yellow (slightly darker)
        color_s3_knn = (1.0, 0.8, 0.3, 0.4)  # Pale Orange Yellow (slightly lighter)

        color_s4_cknn = (0.6, 0.2, 0.2)  # Maroon (slightly darker)
        color_s4_knn = (0.8, 0.4, 0.4, 0.4)  # Light Maroon (slightly lighter)

        color_s5_cknn = (0.0, 0.6, 0.7)  # Dark Turquoise (slightly darker)
        color_s5_knn = (0.4, 0.7, 0.8, 0.4)  # Light Turquoise (slightly lighter)



    # Add numeric values to bars with smaller font and 2 decimal places
    plt.bar_label(plt.bar(br1, S3_knn, color=color_s3_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S3 cKNN np'), label_type='center', fontsize=8, fmt='%.2f')
    plt.bar_label(plt.bar(br2, S3_cknn, color=color_s3_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S3 cKNN p'), label_type='center', fontsize=8, fmt='%.2f')

    plt.bar_label(plt.bar(br3, S4_knn, color=color_s4_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S4 cKNN np'), label_type='center', fontsize=8, fmt='%.2f')
    plt.bar_label(plt.bar(br4, S4_cknn, color=color_s4_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S4 cKNN p'), label_type='center', fontsize=8, fmt='%.2f')

    plt.bar_label(plt.bar(br5, S5_knn, color=color_s5_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S5 cKNN np'), label_type='center', fontsize=8, fmt='%.2f')
    plt.bar_label(plt.bar(br6, S5_cknn, color=color_s5_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S5 cKNN p'), label_type='center', fontsize=8, fmt='%.2f')



    # plt.bar(br1, S3_knn, color=color_s3_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S3 KNN')
    # plt.bar(br2, S3_cknn, color=color_s3_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S3 cKNN')

    # plt.bar(br3, S4_knn, color=color_s4_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S4 KNN')
    # plt.bar(br4, S4_cknn, color=color_s4_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S4 cKNN')

    # plt.bar(br5, S5_knn, color=color_s5_knn, width=barWidth, edgecolor='grey', linewidth=0.25, label = 'S5 KNN')
    # plt.bar(br6, S5_cknn, color=color_s5_cknn, width=barWidth, edgecolor='grey', linewidth=0.25, label='S5 cKNN')

    # Calculate the center positions of each group for x-tick placement
    br_centers = [(a + b) / 2.0 for a, b in zip(br1, br6)]
    # Adding Xticks
    plt.title('Accuracy of CEBRA-KNN decoding across sessions with pupil diameter data' + str(test_type), fontweight='bold', fontsize=10)
    # plt.xlabel('Model', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xticks(br_centers, ['model trained on S3', 'model trained on S4', 'model trained on S5'])
    plt.grid(axis='y', alpha=0.75, linestyle='-', linewidth=0.25)

    # Optimize legend placement (loc='upper left' may work well)
    plt.legend(loc='upper right', bbox_to_anchor=(1.11,1.11), fontsize = 5)
    plt.ylim(0,1)
    plt.savefig('./information/temporary/CKNN_pupilDiam' + str(test_type) + '.png')


def shuffleNeuralData(dataset, indices, animal: str):

    dataset = concatenate_neuraldata(dataset, indices, animal)
    frames = np.arange(0, dataset.shape[0], 1)
    print('frames before shuffling: ', frames)
    random.shuffle(frames)
    print('frames after shuffling: ', frames)
    
    for i in range((dataset.shape[0])):
        dataset[i, :] = dataset[frames[i] ,: ]
    
    return frames, dataset

def shuffleData_1D(dataset, frames):

    for i in range((dataset.shape[0])):
        dataset[i] = dataset[frames[i]]
    
    return dataset


    pass

def getAccuracy(neuraldata_tr, neuraldata_test, ID_tr, ID_test, testID: str):
    
    KNN_model = cebra.KNNDecoder(n_neighbors=36, metric = 'cosine')
    KNN_model.fit(neuraldata_tr, ID_tr)
    accuracy = accuracy_score(ID_test, KNN_model.predict(neuraldata_test))

    print('KNN_decoder_' + testID + '_acc: ', accuracy)

    return accuracy

def trainCebraBehaviour_sh(model, neuraldata_tr, neuraldata_test, behaviourdata_tr, behaviourdata_test, session: int, modelID: str, modeltype: str):

    # same as trainCebraBehaviour but without indices 

    model.fit(neuraldata_tr, behaviourdata_tr)
    
    model_name = "./training_results/" + str(modeltype) + "/cebra_model_" + str(modelID) + ".pt"

    # send model to cpu and save
    model.to('cpu')        
    model.save(model_name)

    text_file_name = "./training_results/" + str(modeltype) + "/training_info_" + str(modelID) + ".txt"

    with open(text_file_name, 'w') as text_file:
        text_file.write("session: " + str(session) + '\n')
        text_file.write("model architecture: " + str(model.model_architecture) + '\n')

    ax2 = cebra.plot_loss(model)
    fig_name = "./training_results/" + str(modeltype) + "/training_loss_" + str(modelID)
    
    plt.savefig(fig_name)

    #plt.show()


    return model