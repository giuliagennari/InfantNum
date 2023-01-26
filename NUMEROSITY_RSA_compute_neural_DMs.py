## [Giulia Gennari 2020/2021] ##
## COMPUTE NEURAL DISSIMILARITY MATRICES AT SEQUENCE OFFSET ##

import numpy as np
import mne 
from scipy.spatial.distance import (pdist, squareform)
from scipy.stats import  pearsonr

def organize_NUM_instr_ids(epochs):
    fSc, fSv  = [18,19,20,21],[22,23,24,25]
    fMc, fMv = [10,11,12,13],[14,15,16,17]
    fLc,fLv = [2,3,4,5],[6,7,8,9]
    tSc,tSv = [42,43,44,45],[46,47,48,49]
    tMc,tMv = [34,35,36,37],[38,39,40,41]
    tLc,tLv = [26,27,28,29],[30,31,32,33]
    groups = [fSc, fSv, fMc, fMv, fLc, fLv, tSc, tSv, tMc, tMv, tLc, tLv]
    new_ids = [111,122, 211,222, 311,322, 411,422, 511,522, 611,622]
    for g in range(len(groups)):
        for id in groups[g]:
            epochs.events[:,2][epochs.events[:,2]==id]=new_ids[g]
    epochs.event_id = {'fSc':111, 'fSv':122,'fMc':211,'fMv':222, 'fLc':311,'fLv':322, 
                       'tSc':411, 'tSv':422,'tMc':511, 'tMv':522,'tLc':611,'tLv':622}
    
    return epochs


epochs_path = ""
RDM_neur_path = ""

subjects = ['S01']

# these are the conditions relevant for the RSA analysis at sequence onset (i.e. ensuring a balanced design)
cond_instr = ['fMc', 'fMv', 'fLc', 'fLv','tSc', 'tSv', 'tMc', 'tMv']

for s in subjects:
    print(" *** SUBJECT "+s+ " *** ")
    epochs = mne.read_epochs(epochs_path+s+"_offset-epo.fif")
    epochs.drop_channels(epochs.info['bads'])
    
    # merge the ids of the 4 notes
    epochs = organize_NUM_instr_ids(epochs)
    print("NUMBER IDS [with instruments separated] set: ", epochs)
                
    # realign epochs on onset of last note
    epochs_S = epochs['tSc','tSv'].crop(-0.08,1.198)
    epochs_S_2 = mne.EpochsArray(epochs_S.get_data(), epochs_S.info, 
                             events= epochs_S.events, tmin=-0.04, event_id = epochs_S.event_id)
    epochs_M = epochs['fMc','fMv','tMc','tMv'].crop(-0.16, 1.118)
    epochs_M_2 = mne.EpochsArray(epochs_M.get_data(), epochs_M.info, 
                             events= epochs_M.events, tmin=-0.04, event_id = epochs_M.event_id)
    epochs_L = epochs['fLc','fLv','tLc','tLv'].crop(-0.4,0.878)
    epochs_L_2 = mne.EpochsArray(epochs_L.get_data(), epochs_L.info, 
                             events= epochs_L.events, tmin=-0.04, event_id = epochs_L.event_id)
    epochs = mne.concatenate_epochs([epochs_S_2, epochs_M_2, epochs_L_2])
    print("EPOCHS have been re-aligned: ", epochs)
    
    epochs.crop(-0.04,1.078) #560 tps
    
    # find least numerous condition
    candidates = ['tSc', 'tSv']
    n_trials = np.array([len(epochs['tSc']),len(epochs['tSv'])])

    for cand in candidates:
        if len(epochs[cand]) == np.min(n_trials):
            least_num_cond = cand
	
    ## DISTANCES ##
    # at each loop, randomly select for each condition the same number of epochs available for the least_num_cond
    
    pearson_distances, spearman_distances  = [],[]
    for perm in range(100):
        # compute evoked responses
        evokeds = []
        for c in cond_instr:
            data = epochs[c].get_data()
            cond_data_ = data.reshape((data.shape[0], len(epochs.ch_names), 4, 140), order='F') # downsample to 125Hz step 1
            cond_data = np.mean(cond_data_, axis=2) # downsample to 125Hz step 2

            indices = np.random.choice(cond_data.shape[0], len(epochs[least_num_cond]), replace=False)
            chosen_data = cond_data[indices]
            ev = np.mean(chosen_data, axis=0)
            evokeds.append(ev)
        
        # compute neural RDMs at each time point
        pearson_DMs_loop = []
        for tp in range(ev.shape[-1]): 

            # 1-Pearson corr
            pearson_matrix = []
            for c1 in range(len(cond_instr)-1):
                for c2 in range(min(c1 + 1, len(cond_instr)-1), len(cond_instr)):
                    corr, pval = pearsonr(np.asarray(evokeds)[c1,:,tp], np.asarray(evokeds)[c2,:,tp])
                    pearson_matrix.append(1-corr)
            pearson_DMs_loop.append(np.asarray(pearson_matrix))
            
        pearson_distances.append(np.asarray(pearson_DMs_loop))
         
    #save neural_DMs
    np.save(RDM_neur_path+s+"_neural_DMs_pearson_125Hz_realigned_8cond", arr=np.asarray(pearson_distances))   

