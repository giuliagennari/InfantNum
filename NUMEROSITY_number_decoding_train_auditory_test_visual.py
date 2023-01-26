## [Giulia Gennari 2020/2021] ##
## [code originally written with python 3.8, mne 0.21 and sklearn 0.23]
# train algorithms incrementally on each schema, then test on visual epochs
# 200 iterations over 3 consecutive fits i.e. one partial_fit for each schema  
# schemas order changes at each cv loop (standardization of test set matches last training set of the cv loop)
# the main outcome is collected in all_scores, the other two score sets are for sanity check analyses

import numpy as np
import mne
import random
from incremental_learning_model import MyBasicModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Vectorizer)

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

def organize_VISUAL_ids(epochs):
    f = [50,51]
    t = [52,53]
    groups = [f,t]
    new_ids = [1, 4]
    for g in range(len(groups)):
        for id in groups[g]:
            epochs.events[:,2][epochs.events[:,2]==id]=new_ids[g]
    epochs.event_id = {'f':1, 't':4}
    
    return epochs

def shuffle_epochs(epochs):
    ep_objects = []
    for ep in epochs.event_id.keys():
        ep_objects.append(epochs[ep])
    
    for ep in range(len(ep_objects)):           
        data = ep_objects[ep].get_data()
        np.random.shuffle(data)
        n_trials = len(ep_objects[ep])
        event_ids = np.repeat(list(ep_objects[ep].event_id.values()), n_trials)
        events  = np.stack((np.arange(n_trials), np.zeros(n_trials, dtype=int), event_ids), axis=1)    
        ep_objects[ep] = mne.EpochsArray(data, info=ep_objects[ep].info, events=events, 
                                          tmin=ep_objects[ep].tmin, event_id=ep_objects[ep].event_id, verbose=0)
    epochs = mne.concatenate_epochs(ep_objects)
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
    print("REGULAR MICRO-AVERAGING COMPLETED")
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

def increase_n_micro_av_trials(indices, n, overlap_limit, factor):
    """ 
    create groups containing n indices with minimal overlap 
    - parameters
    overlap_limit: n indices 2 groups are allowed to share
    factor: increase n groups (i.e. pseudotrials) this number pf times.
    - returns
    list of groups of indices to use for micro-averaging
    - notes
    uncomment #print for sanity check
    
    """
    n_groups = len(indices)//n

    grouped_indices = []
    for gr in range (n_groups):
        grouped_indices.append(indices[gr*n:gr*n+n])

    starting_n_groups = len(grouped_indices)
    while len(grouped_indices) < starting_n_groups*factor:
        new_indices = np.random.permutation(indices) # re-shuffle indices
        ## add new groups of indices ensuring minimal overlap with the exiting ones (i.e. those in grouped_indices)
        for gr in range (n_groups):
            ok = 0
            new_group = new_indices[gr*n:gr*n+n]
            #print(new_group)
            for group in grouped_indices:
                #print(group)
                #print(set(list(group)) & set(list(new_group)))
                if len(set(list(group)) & set(list(new_group))) <= overlap_limit:
                    ok+=1
            #print("current n groups of indices: ", len(grouped_indices))
            #print(ok)
            if ok==len(grouped_indices):
                grouped_indices.append(new_group)
            #print("NEW n groups of indices: ", len(grouped_indices))
            
    return grouped_indices

def micro_average_enhanced(epochs, tmin, n=8, overlap_limit=2, factor=2):
    """ 
    micro-averaging with employment of same trial more than once
    NEEDS the increase_n_micro_av_trials function: 
    - parameters
    n: number of single trials to average together
    overlap_limit: n single trials 2 micro-averaged trials are allowed to share
    factor: increase standard n pseudotrials this number of times.
    tmin: to create new epoch object
    - returns
    micro-averaged epochs object
    
    """
    
    epochs_list = []
    for ep in epochs.event_id.keys():
        epochs_list.append(epochs[ep])
    
    epochs = []
    for ep_obj in epochs_list:
        indices  = increase_n_micro_av_trials(np.arange(len(ep_obj)), n, overlap_limit=overlap_limit, factor=factor)
        
        averaged_data = []
        for group in indices:
            pseudo_trial = np.mean(ep_obj[group].get_data(), axis=0)
            averaged_data.append(pseudo_trial)

        event_ids = np.repeat(list(ep_obj.event_id.values()), len(indices))
        events  = np.stack((np.arange(len(indices)), np.zeros(len(indices), dtype=int), event_ids), axis=1)    
        ep_obj_new = mne.EpochsArray(averaged_data, info=ep_obj.info, events=events, 
                                              tmin=tmin, event_id=ep_obj.event_id)
        epochs.append(ep_obj_new)
    print("MICRO-AVERAGING ENHANCED COMPLETED")
    return mne.concatenate_epochs(epochs)


epochs_path = ""
path_scores = ""

subjects = ['S01']

pipe_preprocess = make_pipeline(Vectorizer(), StandardScaler())
prepr_slider = SlidingEstimator(pipe_preprocess, n_jobs=10)

for s in subjects:
    print("\n *** SUBJECT "+s+ " *** \n")
    # auditory epochs
    epochs = mne.read_epochs(epochs_path+s[:3]+"_offset-epo.fif", proj=False, preload = False)
    epochs.detrend = 1
    epochs.load_data()
    # visual epochs
    visual_epochs = mne.read_epochs(epochs_path+s+"_visual-epo.fif", proj=False, preload = False)
    visual_epochs.detrend = 1
    visual_epochs.load_data()

    ## match bad channels: list of bad channels needs to overlap
    print("bad channels at start: ")
    print("offset epochs: " , epochs.info['bads'])
    print("visual epochs: " , visual_epochs.info['bads'])
    
    for bad_ch in epochs.info['bads']:
        if bad_ch not in visual_epochs.info['bads']:
            visual_epochs.info['bads'] = visual_epochs.info['bads'] + [bad_ch]
            
    for bad_ch in visual_epochs.info['bads']:
        if bad_ch not in epochs.info['bads']:
            epochs.info['bads'] = epochs.info['bads'] + [bad_ch]
    
    print("bad channels after matching: ")
    print("offset epochs: " , sorted(epochs.info['bads']))
    print("visual epochs: " , sorted(visual_epochs.info['bads']))
    
    # once common bads are set you can drop them and apply eeg reference 
    epochs.drop_channels(epochs.info['bads'])
    visual_epochs.drop_channels(visual_epochs.info['bads'])
    
    epochs.set_eeg_reference(projection=False)
    visual_epochs.set_eeg_reference(projection=False)
    # crop visual epochs to match lenght of auditory offset epochs
    visual_epochs.crop(-0.04,1.238)
    
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
    
    print("\n - SCHEMA A: prepare data - \n")
    
    epochs_ = epochs.copy()
    epochs_.equalize_event_counts(['fM', 'fL']) # this is why you need to work on a copy of the original epochs
    XA_four_M = epochs_['fM'].get_data()
    yA_four_M = epochs_['fM'].events[:,2]
    XA_four_L = epochs_['fL'].get_data()
    yA_four_L = epochs_['fL'].events[:,2]
    
    epochs_trainingA_twelve = epochs_['tM']
    
    cvA = ShuffleSplit(100, test_size=0.25)
    train_indices_A, test_indices_A = [],[]
    for train_index , test_index in cvA.split(XA_four_M, yA_four_M):
        train_indices_A.append(train_index)
        test_indices_A.append(test_index)
 
    print("\n - SCHEMA B: prepare data - \n")

    XB_twelve_S = epochs['tS'].get_data()
    yB_twelve_S = epochs['tS'].events[:,2] 
    cvS = ShuffleSplit(100, test_size=0.2)
    train_indices_B_tS, test_indices_B_tS = [], []
    for train_index , test_index in cvS.split(XB_twelve_S, yB_twelve_S):
        train_indices_B_tS.append(train_index)
        test_indices_B_tS.append(test_index)

    XB_twelve_M = epochs['tM'].get_data()
    yB_twelve_M = epochs['tM'].events[:,2]
    n_trials_tM_test = len(epochs['tM'])- len(train_indices_B_tS[0])
    cvM = ShuffleSplit(100, test_size=n_trials_tM_test)
    train_indices_B_tM, test_indices_B_tM = [], []
    for train_index , test_index in cvM.split(XB_twelve_M, yB_twelve_M):
        train_indices_B_tM.append(train_index)
        test_indices_B_tM.append(test_index)

    epochs_trainingB_four = epochs['fM']
    
    print("\n - SCHEMA C: prepare data - \n")

    XC_twelve_L = epochs['tL'].get_data()
    yC_twelve_L = epochs['tL'].events[:,2]
    cvL = ShuffleSplit(100, test_size=0.2)
    train_indices_C_tL, test_indices_tL = [], []
    for train_index , test_index in cvL.split(XC_twelve_L, yC_twelve_L):
        train_indices_C_tL.append(train_index)
        test_indices_tL.append(test_index)

    XC_twelve_M = epochs['tM'].get_data()
    yC_twelve_M = epochs['tM'].events[:,2]
    n_trials_tM_test = len(epochs['tM'])- len(train_indices_C_tL[0])
    cvM = ShuffleSplit(100, test_size=n_trials_tM_test)
    train_indices_C_tM, test_indices_tM = [], []
    for train_index , test_index in cvM.split(XC_twelve_M, yC_twelve_M):
        train_indices_C_tM.append(train_index)
        test_indices_tM.append(test_index)
    
    epochs_trainingC_four = epochs['fL']
    
    ### CV LOOPS FINALLY! ###
    all_scores, all_scores_e, all_scores_i = [], [], []
    for loop in range(100):
        
        # clean the model
        model = MyBasicModel(loss='log', average=True )
        time_gen = GeneralizingEstimator(model, scoring='roc_auc', n_jobs=14)
        # print("\n new model initiated for new CV LOOP: \n", time_gen.base_estimator['mybasicmodel'])
        
        ### training set for schema A
        XA_train_four_M_, XA_test_four_M_ = XA_four_M[train_indices_A[loop]], XA_four_M[test_indices_A[loop]]
        XA_train_four_M = micro_average_within_cv(data = XA_train_four_M_ , n=8)
        yA_train_four_M = np.repeat(1, len(XA_train_four_M))

        XA_train_four_L_, XA_test_four_L_ = XA_four_L[train_indices_A[loop]], XA_four_L[test_indices_A[loop]]
        XA_train_four_L = micro_average_within_cv(data = XA_train_four_L_ , n=8)
        yA_train_four_L = np.repeat(1, len(XA_train_four_L))
        
        XA_train_twelve = micro_average_within_cv(data = epochs_trainingA_twelve.get_data(), n=8)
        yA_train_twelve = np.repeat(4, len(XA_train_twelve))

        ## finalize train set
        XA_train_temp = np.vstack((XA_train_four_M, XA_train_four_L, XA_train_twelve))
        XA_train = XA_train_temp.reshape((XA_train_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("XA_train final shape: ", XA_train.shape)
        yA_train = np.concatenate((yA_train_four_M, yA_train_four_L, yA_train_twelve))
        # compute weights
        weights_ = compute_class_weight('balanced', np.unique(yA_train), yA_train)
        weights_A = {1:weights_[0],4:weights_[1]}
        
        ### training set for schema schema B
        XB_train_twelve_S_, XB_test_twelve_S_ = XB_twelve_S[train_indices_B_tS[loop]], XB_twelve_S[test_indices_B_tS[loop]]
        XB_train_twelve_S = micro_average_within_cv(data = XB_train_twelve_S_ , n=8)
        yB_train_twelve_S = np.repeat(4, len(XB_train_twelve_S))

        XB_train_twelve_M_, XB_test_twelve_M_ = XB_twelve_M[train_indices_B_tM[loop]], XB_twelve_M[test_indices_B_tM[loop]]
        XB_train_twelve_M = micro_average_within_cv(data = XB_train_twelve_M_ , n=8)
        yB_train_twelve_M = np.repeat(4, len(XB_train_twelve_M))
        
        XB_train_four = micro_average_within_cv(data = epochs_trainingB_four.get_data(), n=8)
        yB_train_four = np.repeat(1, len(XB_train_four))
        
        ## finalize train set
        XB_train_temp = np.vstack(( XB_train_twelve_S, XB_train_twelve_M, XB_train_four))
        XB_train = XB_train_temp.reshape((XB_train_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("XB_train final shape: ", XB_train.shape)
        yB_train = np.concatenate((yB_train_twelve_S, yB_train_twelve_M, yB_train_four))
        # compute weights
        weights_ = compute_class_weight('balanced', np.unique(yB_train), yB_train)
        weights_B = {1:weights_[0],4:weights_[1]}
       
        ### training set for schema C
        XC_train_twelve_L_, XC_test_twelve_L_ = XC_twelve_L[train_indices_C_tL[loop]], XC_twelve_L[test_indices_tL[loop]]
        XC_train_twelve_L = micro_average_within_cv(data = XC_train_twelve_L_ , n=8)
        yC_train_twelve_L = np.repeat(4, len(XC_train_twelve_L))

        XC_train_twelve_M_, XC_test_twelve_M_ = XC_twelve_M[train_indices_C_tM[loop]], XC_twelve_M[test_indices_tM[loop]]
        XC_train_twelve_M = micro_average_within_cv(data = XC_train_twelve_M_ , n=8)
        yC_train_twelve_M = np.repeat(4, len(XC_train_twelve_M))
        
        XC_train_four = micro_average_within_cv(data = epochs_trainingC_four.get_data(), n=8)
        yC_train_four = np.repeat(1, len(XC_train_four))
        
        ## finalize train set
        XC_train_temp = np.vstack(( XC_train_twelve_L, XC_train_twelve_M, XC_train_four))
        XC_train = XC_train_temp.reshape((XC_train_temp.shape[0], len(epochs.ch_names), 5, int(640/5)) ,order='F')
        print ("XC_train final shape: ", XC_train.shape)
        yC_train = np.concatenate((yC_train_twelve_L, yC_train_twelve_M, yC_train_four))
        # compute weights
        weights_ = compute_class_weight('balanced', np.unique(yC_train), yC_train)
        weights_C = {1:weights_[0], 4:weights_[1]}
                
        A = [weights_A, XA_train, yA_train]
        B = [weights_B, XB_train, yB_train]
        C = [weights_C, XC_train, yC_train]
        schemas_list = [A,B,C]
        random.shuffle(schemas_list) # so that: different training order at each cv loop
        
        # vectorize and standardize every train set 
        for scm in schemas_list:
            scm[1] = prepr_slider.fit_transform(scm[1], scm[2])
            scm[1] = np.swapaxes(scm[1],1,2)
        # fitting in multiple iterations 
        for i in range(200):
            for scm in schemas_list:
                weights, X_train, y_train = scm[0], scm[1], scm[2]
                model.class_weight = weights
                time_gen.fit(X_train, y_train)            
        
        # prepare VISUAL test sets and score
        test_epochs = shuffle_epochs(visual_epochs) # shuffle at each cv run so that different trials are discarded by the equalization below
        test_epochs.equalize_event_counts(['v04e','v04i'], method='truncate')
        test_epochs.equalize_event_counts(['v12e','v12i'], method='truncate')
        test_epochs = organize_VISUAL_ids(test_epochs)
        n_final_pseudotrials_check = [len(test_epochs['f'])//8<5, len(test_epochs['t'])//8<5]
        if True in n_final_pseudotrials_check:
            test_epochs = micro_average_enhanced(test_epochs, tmin=0., overlap_limit=2, factor=1.5)
        else:
            test_epochs = micro_average(test_epochs, n=8)
        test_epochs.equalize_event_counts(test_epochs.event_id)
        print("perfectly balanced visual epochs test set: ", test_epochs)
        
        X_test = test_epochs.get_data().reshape((len(test_epochs), len(visual_epochs.ch_names), 5, int(640/5)) ,order='F')
        print("test set across reshaped: ", X_test.shape)
        y_test = test_epochs.events[:,2]
        
        X_test = prepr_slider.transform(X_test)
        X_test = np.swapaxes(X_test,1,2)
        print("test set transformed: ", X_test.shape)

        scores = time_gen.score(X_test, y_test)
        all_scores.append(scores) 
        
        ### test on trials with extensive/intensive parameter control separately 

        test_epochs_e = visual_epochs['v04e', 'v12e']
        test_epochs_i = visual_epochs['v04i', 'v12i']
        print("test epochs extensive constant selected:", test_epochs_e)
        print("test epochs intensive constant selected:", test_epochs_i)        

        test_epochs_controlled = [test_epochs_e, test_epochs_i]
        test_epochs_controlled_ready = []
        for ep in test_epochs_controlled:
            ep = organize_VISUAL_ids(ep)
            n_final_pseudotrials_check = [len(ep['f'])//8<3, len(ep['t'])//8<3]
            if True in n_final_pseudotrials_check:
                ep = micro_average_enhanced(ep, tmin=0., overlap_limit=4, factor=2)
            else:
                ep = micro_average(ep, n=8)
            ep.equalize_event_counts(ep.event_id)
            test_epochs_controlled_ready.append(ep)

        test_epochs_e, test_epochs_i = test_epochs_controlled_ready
        print("test epochs extensive constant ready:", test_epochs_e)
        print("test epochs intensive constant ready:", test_epochs_i)        
        
        # proceed with test "extensive constant"
        X_test_e = test_epochs_e.get_data().reshape((len(test_epochs_e), len(test_epochs_e.ch_names), 5, int(640/5)) ,order='F')
        print("test set across reshaped: ", X_test_e.shape)
        y_test_e = test_epochs_e.events[:,2]
        # standardize
        X_test_e = prepr_slider.transform(X_test_e)
        X_test_e = np.swapaxes(X_test_e,1,2)
        print("test set transformed: ", X_test_e.shape)
        # score 
        scores_e = time_gen.score(X_test_e, y_test_e)
        all_scores_e.append(scores_e)  
        
        # proceed with test "intensive constant"
        X_test_i = test_epochs_i.get_data().reshape((len(test_epochs_i), len(test_epochs_i.ch_names), 5, int(640/5)) ,order='F')
        print("test set across reshaped: ", X_test_i.shape)
        y_test_i = test_epochs_i.events[:,2]
        # standardize
        X_test_i = prepr_slider.transform(X_test_i)
        X_test_i = np.swapaxes(X_test_i,1,2)
        print("test set transformed: ", X_test_i.shape)
        # score 
        scores_i = time_gen.score(X_test_i, y_test_i)
        all_scores_i.append(scores_i)          

        np.save(path_scores+s+"_number_incremental_ASGD_test_VISUAL_balanced_500Hz_tp5_av8_L2", arr = np.asarray(all_scores))
        np.save(path_scores+s+"_number_incremental_ASGD_test_VISUAL_e_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_e))
        np.save(path_scores+s+"_number_incremental_ASGD_test_VISUAL_i_500Hz_tp5_av8_L2", arr = np.asarray(all_scores_i))
