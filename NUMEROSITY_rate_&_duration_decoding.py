## [Giulia Gennari 2020/2021] ##
## train decoders to check whether main results are due tue rate or duration learning ##
## [code originally written with python 3.8, mne 0.21 and sklearn 0.23]

import numpy as np
import mne
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Vectorizer)

epochs_path = ""
path_scores = ""

def organize_NUM_ids(epochs):
    fS = [18,19,20,21,22,23,24,25]
    fM = [10,11,12,13,14,15,16,17]
    fL = [2,3,4,5,6,7,8,9]
    tS = [42,43,44,45,46,47,48,49]
    tM = [34,35,36,37,38,39,40,41]
    tL = [26,27,28,29,30,31,32,33]
    groups = [fL, fS, fM, tS, tM, tL]
    new_ids = [3,1,2,4,5,6]
    for g in range(len(groups)):
        for id in groups[g]:
            epochs.events[:,2][epochs.events[:,2]==id]=new_ids[g]
    epochs.event_id = {'fS':1, 'fM':2, 'fL':3, 'tS':4, 'tM':5, 'tL':6}
    
    return epochs

def micro_average(epochs, n):
    nb_trial_to_average = n
    ep_objects = []
    for ep in epochs.event_id.keys():
        ep_objects.append(epochs[ep])
    
    for ep in range(len(ep_objects)):
        nb_averaged_trials = len(ep_objects[ep])//nb_trial_to_average            
        data = ep_objects[ep].get_data()
        np.random.shuffle(data)
        data_averaged = []
        for itrial_ave in range(nb_averaged_trials):
            average = np.mean(data[itrial_ave*nb_trial_to_average:itrial_ave*nb_trial_to_average+nb_trial_to_average], axis=0)
            data_averaged.append(average)

        event_ids = np.repeat(list(ep_objects[ep].event_id.values()), nb_averaged_trials)
        events  = np.stack((np.arange(nb_averaged_trials),np.zeros(nb_averaged_trials, dtype=int), event_ids), axis=1)    
        ep_objects[ep] = mne.EpochsArray(data_averaged, info=ep_objects[ep].info, events=events, 
                                          tmin=ep_objects[ep].tmin, event_id=ep_objects[ep].event_id)

    epochs = mne.concatenate_epochs(ep_objects)
    return epochs


subjects = ['S01',]

cv = ShuffleSplit(100, test_size=0.15)
clf = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(solver='liblinear'))
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=14)


for s in subjects:
    print(" *** SUBJECT "+s+ " *** ")
    epochs = mne.read_epochs(epochs_path+s+"_offset-epo.fif", proj=False, preload = False)
    epochs.detrend = 1
    epochs.load_data()
    epochs.drop_channels(epochs.info['bads'])
    epochs.set_eeg_reference(projection=False)
    
    epochs = organize_NUM_ids(epochs)
    print("NUMBER IDS set: ", epochs)
    
    # realign epochs on onset of last note
    # (note: EpochsArray necessary to re-set tmin)
    epochs_S = epochs['tS'].crop(-0.08,1.198)
    epochs_S = mne.EpochsArray(epochs_S.get_data(), epochs_S.info, 
                             events= epochs_S.events, tmin=-0.04, event_id = epochs_S.event_id)
    epochs_M = epochs['fM','tM'].crop(-0.16, 1.118)
    epochs_M = mne.EpochsArray(epochs_M.get_data(), epochs_M.info, 
                             events= epochs_M.events, tmin=-0.04, event_id = epochs_M.event_id)
    epochs_L = epochs['fL','tL'].crop(-0.4,0.878)
    epochs_L = mne.EpochsArray(epochs_L.get_data(), epochs_L.info, 
                             events= epochs_L.events, tmin=-0.04, event_id = epochs_L.event_id)
    epochs = mne.concatenate_epochs([epochs_S, epochs_M, epochs_L])
    print("EPOCHS have been re-aligned: ", epochs)

    ### DECODING  RATE ###
    print(" - DECODE RATE: 1.9 vs 5.6 Hz - ") 
    epochs_training = epochs['fL', 'tM']
    
    all_scores_within, all_scores = [], []
    for train_index , test_index in cv.split(epochs_training.get_data(), epochs_training.events[:,2]):
        # split
        epochs_train, epochs_test_within = epochs_training[train_index], epochs_training[test_index]
        # micro-average
        epochs_train = micro_average(epochs_train, n=8)
        epochs_test_within = micro_average(epochs_test_within, n=8)
        # equalize event counts
        epochs_train.equalize_event_counts(epochs_train.event_id)
        epochs_test_within.equalize_event_counts(epochs_test_within.event_id)
        
        # prepare data for decoders
        X_train_temp = epochs_train.get_data()
        X_train = X_train_temp.reshape((X_train_temp.shape[0], len(epochs.ch_names), 5, int(X_train_temp.shape[-1]/5)) ,order='F')
        print ("X_train final shape: ", X_train.shape)
        y_train = epochs_train.events[:,2]
        #fit
        time_gen.fit(X_train, y_train)
        
        #test within the same set
        X_test_within_temp = epochs_test_within.get_data()
        X_test_within = X_test_within_temp.reshape((X_test_within_temp.shape[0], len(epochs.ch_names), 5, int(X_test_within_temp.shape[-1]/5)) ,order='F')
        print ("X_test_within final shape: ", X_test_within.shape)
        y_test_within = epochs_test_within.events[:,2]

        scores_within = time_gen.score(X_test_within, y_test_within)
        all_scores_within.append(scores_within)
        
        #create data set to test on unseen rates
        epochs_test = epochs['tL', 'fM']
        # match event ids with training set
        epochs_test.events[:,2][epochs_test.events[:,2]==6]=3
        epochs_test.events[:,2][epochs_test.events[:,2]==2]=5
        epochs_test.event_id = {'tL':3, 'fM':5}

        epochs_test = micro_average(epochs_test, n=8)
        epochs_test.equalize_event_counts(epochs_test.event_id)
        print("epochs to test on new rates ready: ", epochs_test)
        
        X_test_temp = epochs_test.get_data()
        X_test = X_test_temp.reshape((X_test_temp.shape[0], len(epochs.ch_names), 5, int(X_test_temp.shape[-1]/5)) ,order='F')
        print ("X_test final shape: ", X_test.shape)
        y_test = epochs_test.events[:,2]
        
        scores = time_gen.score(X_test, y_test)
        all_scores.append(scores)
        
    np.save(path_scores+s+"_rate_test_within_REALIGNED_detr_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_within))
    np.save(path_scores+s+"_rate_test_new_durations_REALIGNED_detr_500Hz_tp5_av8_L2", arr = np.asarray(all_scores))   

    ### DECODING  DURATION ###
    print(" - DECODE DURATION: 720 vs 2160 ms - ") 
    epochs_training = epochs['fM', 'tM']
    
    all_scores_within, all_scores = [], []
    for train_index , test_index in cv.split(epochs_training.get_data(), epochs_training.events[:,2]):
        # split
        epochs_train, epochs_test_within = epochs_training[train_index], epochs_training[test_index]
        # micro-average
        epochs_train = micro_average(epochs_train, n=8)
        epochs_test_within = micro_average(epochs_test_within, n=8)
        # equalize event counts
        epochs_train.equalize_event_counts(epochs_train.event_id)
        epochs_test_within.equalize_event_counts(epochs_test_within.event_id)
        
        # prepare data for decoders
        X_train_temp = epochs_train.get_data()
        X_train = X_train_temp.reshape((X_train_temp.shape[0], len(epochs.ch_names), 5, int(X_train_temp.shape[-1]/5)), order='F')
        print ("X_train final shape: ", X_train.shape)
        y_train = epochs_train.events[:,2]
        #fit
        time_gen.fit(X_train, y_train)
        
        #test within the same set
        X_test_within_temp = epochs_test_within.get_data()
        X_test_within = X_test_within_temp.reshape((X_test_within_temp.shape[0], len(epochs.ch_names), 5, int(X_test_within_temp.shape[-1]/5)) ,order='F')
        print ("X_test_within final shape: ", X_test_within.shape)
        y_test_within = epochs_test_within.events[:,2]

        scores_within = time_gen.score(X_test_within, y_test_within)
        all_scores_within.append(scores_within)
        
        #create data set to test on unseen rates
        epochs_test = epochs['tS', 'fL']
        epochs_test.events[:,2][epochs_test.events[:,2]==4]=2
        epochs_test.events[:,2][epochs_test.events[:,2]==3]=5
        epochs_test.event_id = {'tS':2, 'fL':5}

        epochs_test = micro_average(epochs_test, n=8)
        epochs_test.equalize_event_counts(epochs_test.event_id)
        print("epochs to test on new rates ready: ", epochs_test)
        
        X_test_temp = epochs_test.get_data()
        X_test = X_test_temp.reshape((X_test_temp.shape[0], len(epochs.ch_names), 5, int(X_test_temp.shape[-1]/5)) ,order='F')
        print ("X_test final shape: ", X_test.shape)
        y_test = epochs_test.events[:,2]
        
        scores = time_gen.score(X_test, y_test)
        all_scores.append(scores)
        
    np.save(path_scores+s+"_duration_test_within_REALIGNED_detr_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_within))
    np.save(path_scores+s+"_duration_test_new_rates_REALIGNED_detr_500Hz_tp5_av8_L2", arr = np.asarray(all_scores))   
    
