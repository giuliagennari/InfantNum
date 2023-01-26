## [Giulia Gennari 2020/2021] ##
## DECODING NUMBERS ON AUDITORY TRIALS ##
## [code originally written with python 3.8, mne 0.21 and sklearn 0.23]

import numpy as np
import mne

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from mne.decoding import ( GeneralizingEstimator, Vectorizer)

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

def micro_average_within_cv(data, n):
    nb_trial_to_average = n
    print("data shape before micro-averaging: ", np.asarray(data).shape)
    
    nb_averaged_trials = len(data)//nb_trial_to_average
    np.random.shuffle(data)
    data_averaged = []
    for itrial_ave in range(nb_averaged_trials):
        average = np.mean(data[itrial_ave*nb_trial_to_average:itrial_ave*nb_trial_to_average+nb_trial_to_average], axis=0)
        data_averaged.append(average)
    print("data shape after micro-averaging: ", np.asarray(data_averaged).shape)    

    return data_averaged

subjects = ['S01']

clf = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(solver= 'liblinear', class_weight='balanced'))
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=14)

### test all schemas with GeneralizingEstimator after re-alignement on last note onsets 
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
        
    ### DECODING SCHEMA: A ###

    print(" - SCHEMA A in cv - ")
    epochs_ = epochs.copy()

    epochs_.equalize_event_counts(['fM', 'fL'])

    X_four_M = epochs_['fM'].get_data()
    y_four_M = epochs_['fM'].events[:,2]

    X_four_L = epochs_['fL'].get_data()
    y_four_L = epochs_['fL'].events[:,2]
    
    epochs_training_twelve = epochs_['tM']
    
    cvA = ShuffleSplit(100, test_size=0.15)
    all_scores_A_test_1, all_scores_A_test_2 = [], []
    for train_index , test_index in cvA.split(X_four_M, y_four_M):

        X_train_four_M_, X_test_four_M_ = X_four_M[train_index], X_four_M[test_index]
        X_train_four_M = micro_average_within_cv(data = X_train_four_M_ , n=8)
        y_train_four_M = np.repeat(1, len(X_train_four_M))
        X_test_four_M = micro_average_within_cv(data = X_test_four_M_ , n=8)
        y_test_four_M = np.repeat(1, len(X_test_four_M))

        X_train_four_L_, X_test_four_L_ = X_four_L[train_index], X_four_L[test_index]
        X_train_four_L = micro_average_within_cv(data = X_train_four_L_ , n=8)
        y_train_four_L = np.repeat(1, len(X_train_four_L))
        X_test_four_L = micro_average_within_cv(data = X_test_four_L_ , n=8)
        y_test_four_L = np.repeat(1, len(X_test_four_L))
        
        X_train_twelve = micro_average_within_cv(data = epochs_training_twelve.get_data(), n=8)
        y_train_twelve = np.repeat(4, len(X_train_twelve))

        X_train_temp = np.vstack((X_train_four_M, X_train_four_L, X_train_twelve))
        X_train = X_train_temp.reshape((X_train_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("X_train final shape: ", X_train.shape)
        y_train = np.concatenate((y_train_four_M, y_train_four_L, y_train_twelve))

        time_gen.fit(X_train, y_train)

        # test on duration matched trials
        X_test_twelve_1 = micro_average_within_cv(data = epochs_['tS'].get_data(), n=8)
        twelve_indices_1 = np.random.choice(len(X_test_twelve_1), len(X_test_four_M), replace=False)
        print("indices to select among twelve S trials: ", twelve_indices_1)
        X_test_1_temp = np.vstack((X_test_four_M, np.asarray(X_test_twelve_1)[twelve_indices_1]))
        X_test_1 = X_test_1_temp.reshape((X_test_1_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("X_test 1 (duration match) final shape: ", X_test_1.shape)
        y_test_1 = np.concatenate((y_test_four_M, np.repeat(4, len(twelve_indices_1))))
        scores_test_1 = time_gen.score(X_test_1, y_test_1)
        all_scores_A_test_1.append(scores_test_1)

        # test on rate matched trials
        X_test_twelve_2 = micro_average_within_cv(data = epochs_['tL'].get_data(), n=8)
        twelve_indices_2 = np.random.choice(len(X_test_twelve_2), len(X_test_four_L), replace=False)
        print("indices to select among twelve L trials: ", twelve_indices_2)
        X_test_2_temp = np.vstack((X_test_four_L, np.asarray(X_test_twelve_2)[twelve_indices_2]))
        X_test_2 = X_test_2_temp.reshape((X_test_2_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("X_test 2 (rate match) final shape: ", X_test_2.shape)
        y_test_2 = np.concatenate((y_test_four_L, np.repeat(4, len(twelve_indices_2))))
        scores_test_2 = time_gen.score(X_test_2, y_test_2)
        all_scores_A_test_2.append(scores_test_2)
    
        
    np.save(path_scores+s+"_schema_A_test_duration_matched_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_A_test_1))
    np.save(path_scores+s+"_schema_A_test_rate_matched_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_A_test_2))   

    ### DECODING SCHEMA: B ###

    print(" - SCHEMA B in cv - ")
    #epochs_ = epochs.copy()

    X_twelve_S = epochs['tS'].get_data()
    y_twelve_S = epochs['tS'].events[:,2]
    cvS = ShuffleSplit(100, test_size=0.2)
    train_indices_tS, test_indices_tS = [], []
    for train_index , test_index in cvS.split(X_twelve_S, y_twelve_S):
        train_indices_tS.append(train_index)
        test_indices_tS.append(test_index)

    X_twelve_M = epochs['tM'].get_data()
    y_twelve_M = epochs['tM'].events[:,2]

    n_trials_tM_test = len(epochs['tM'])- len(train_indices_tS[0])
    
    cvM = ShuffleSplit(100, test_size=n_trials_tM_test)
    train_indices_tM, test_indices_tM = [], []
    for train_index , test_index in cvM.split(X_twelve_M, y_twelve_M):
        train_indices_tM.append(train_index)
        test_indices_tM.append(test_index)

    epochs_training_four = epochs['fM']

    all_scores_B_test_1, all_scores_B_test_2 = [],[]
    for loop in range(100):
        X_train_twelve_S_, X_test_twelve_S_ = X_twelve_S[train_indices_tS[loop]], X_twelve_S[test_indices_tS[loop]]
        X_train_twelve_S = micro_average_within_cv(data = X_train_twelve_S_ , n=8)
        y_train_twelve_S = np.repeat(4, len(X_train_twelve_S))
        X_test_twelve_S = micro_average_within_cv(data = X_test_twelve_S_ , n=8)
        y_test_twelve_S = np.repeat(4, len(X_test_twelve_S))

        X_train_twelve_M_, X_test_twelve_M_ = X_twelve_M[train_indices_tM[loop]], X_twelve_M[test_indices_tM[loop]]
        X_train_twelve_M = micro_average_within_cv(data = X_train_twelve_M_ , n=8)
        y_train_twelve_M = np.repeat(4, len(X_train_twelve_M))
        X_test_twelve_M = micro_average_within_cv(data = X_test_twelve_M_ , n=8)
        y_test_twelve_M = np.repeat(4, len(X_test_twelve_M))
        
        X_train_four = micro_average_within_cv(data = epochs_training_four.get_data(), n=8)
        y_train_four = np.repeat(1, len(X_train_four))

        X_train_temp = np.vstack(( X_train_twelve_S, X_train_twelve_M, X_train_four))
        X_train = X_train_temp.reshape((X_train_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("X_train final shape: ", X_train.shape)
        y_train = np.concatenate((y_train_twelve_S, y_train_twelve_M, y_train_four))

        time_gen.fit(X_train, y_train)

        # test on duration matched trials
        X_test_four_1 = micro_average_within_cv(data = epochs['fL'].get_data(), n=8)
        four_indices_1 = np.random.choice(len(X_test_four_1), len(X_test_twelve_M), replace=False)
        print("indices to select among four L trials: ", four_indices_1)
        X_test_1_temp = np.vstack((X_test_twelve_M, np.asarray(X_test_four_1)[four_indices_1]))
        X_test_1 = X_test_1_temp.reshape((X_test_1_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("X_test 1 (duration match) final shape: ", X_test_1.shape)
        y_test_1 = np.concatenate((y_test_twelve_M, np.repeat(1, len(four_indices_1))))
        scores_test_1 = time_gen.score(X_test_1, y_test_1)
        all_scores_B_test_1.append(scores_test_1)

        # test rate matched trials
        epochs_test_2 = epochs['fL','tL']
        epochs_test_2.events[:,2][epochs_test_2.events[:,2]==3]=1
        epochs_test_2.events[:,2][epochs_test_2.events[:,2]==6]=4
        epochs_test_2.event_id = {'f':1, 't':4}
		
        epochs_test_2 = micro_average(epochs_test_2, n=8)
        epochs_test_2.equalize_event_counts(epochs_test_2.event_id)
        print('TEST SET rate matched - long notes: ', epochs_test_2)

        X_test_temp = epochs_test_2.get_data()
        X_test = X_test_temp.reshape((len(epochs_test_2), len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print("dataset reshaped for test 2 (rate match)", X_test.shape) # give estimators 5 time points at a time i.e. 10ms
        y_test = epochs_test_2.events[:, 2]

        scores_2 = time_gen.score(X_test, y_test)
        all_scores_B_test_2.append(scores_2)

    
    np.save(path_scores+s+"_schema_B_test_duration_matched_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_B_test_1))
    np.save(path_scores+s+"_schema_B_test_rate_matched_optionB_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_B_test_2))	

    
    ### DECODING SCHEMA: C ###

    print(" - SCHEMA C - micro-average at each cv loop ")
    
    n_trials_tM_test = len(epochs['tM'])-len(epochs['tL'])
    #make sure test set doesn't contain less than 15% of tM epochs
    check = n_trials_tM_test*100//len(epochs['tM'])
    if check<15:
        n_trials_tM_test = 0.15
        
    cvC = ShuffleSplit(100, test_size=n_trials_tM_test)
    
    X_twelve_M = epochs['tM'].get_data()
    y_twelve_M = epochs['tM'].events[:,2]
    
    all_scores_dur_matched, all_scores_rate_matched = [],[]
    for train_index , test_index in cvC.split(X_twelve_M, y_twelve_M ):
        X_train_twelve_M, X_test_twelve_M = X_twelve_M[train_index], X_twelve_M[test_index]
        epochs_twelve_M_train = mne.EpochsArray(X_train_twelve_M, epochs['tM'].info,events= epochs['tM'].events[train_index],
                                                tmin= epochs['tM'].tmin, event_id = epochs['tM'].event_id)
        epochs_ = mne.concatenate_epochs([epochs['fS', 'fM', 'fL', 'tS','tL'], epochs_twelve_M_train])
        epochs_ = micro_average(epochs_, n=8)
        
        epochs_.equalize_event_counts(['tM', 'tL']) # this should not drop anything unless test_size=0.15 is used by cv
        epochs_training = epochs_['fL','tM', 'tL']
        epochs_training.events[:,2][epochs_training.events[:,2]==3]=1
        epochs_training.events[:,2][epochs_training.events[:,2]==5]=4
        epochs_training.events[:,2][epochs_training.events[:,2]==6]=4
        epochs_training.event_id = {'f':1, 't':4}
        print( "epochs for training are ready: ", epochs_training)
        
        # first fit
        X_train_temp = epochs_training.get_data()
        X_train = X_train_temp.reshape((len(epochs_training), len(epochs_.ch_names), 5, int(640/5)) ,order='F')
        print("dataset reshaped for first training ", X_train.shape) # give estimators 5 time points at a time i.e. 10ms
        y_train = epochs_training.events[:, 2]

        time_gen.fit(X_train, y_train)
        
        #test duration matched 
        epochs_test_1 = epochs_['fM','tS']
        epochs_test_1.equalize_event_counts(epochs_test_1.event_id)
        epochs_test_1.events[:,2][epochs_test_1.events[:,2]==2]=1
        epochs_test_1.event_id = {'fM':1, 'tS':4}
        print('TEST SET duration matched: ', epochs_test_1)
        
        X_test_temp = epochs_test_1.get_data()
        X_test = X_test_temp.reshape((len(epochs_test_1), len(epochs_.ch_names), 5, int(640/5)) ,order='F')
        print("dataset reshaped for test 1 (duration matched)", X_test.shape) # give estimators 5 time points at a time i.e. 10ms
        y_test = epochs_test_1.events[:, 2]

        scores_1 = time_gen.score(X_test, y_test)
        all_scores_dur_matched.append(scores_1)
        
        # test on rate matched -medium notes
        X_test_twelve = micro_average_within_cv(data = X_test_twelve_M, n=8)
        four_indices = np.random.choice(len(epochs_['fM']), len(X_test_twelve), replace=False)
        print("indices to select among four M trials: ", four_indices)
        X_test_four = epochs_['fM'].get_data()[four_indices]
        
        X_test_2_temp = np.vstack((X_test_four, X_test_twelve))
        X_test_2 = X_test_2_temp.reshape((X_test_2_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("X_test 2 (rate match option B) final shape: ", X_test_2.shape)
        y_test_2 = np.concatenate((np.repeat(1, len(X_test_four)), np.repeat(4, len(X_test_twelve))))
        
        scores_2 = time_gen.score(X_test_2, y_test_2)
        all_scores_rate_matched.append(scores_2)
        
    np.save(path_scores+s+"_schema_C_test_duration_matched_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_dur_matched))
    np.save(path_scores+s+"_schema_C_test_rate_matched_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_rate_matched))