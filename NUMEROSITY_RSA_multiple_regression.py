## [Giulia Gennari 2020/2021] ##
## RSA step 3: perform LINEAR REGRESSION i.e. compute beta coefficients at single subject level

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


neural_DMs_path = ""
predictor_DM_path = ""
save_path = ""

subjects = ['S01']

LR = LinearRegression(n_jobs=2)

# prepare predictors 
num_dist = np.load(predictor_DM_path+"number_DM_8cond.npy")
rate_dist = np.load(predictor_DM_path+"rate_DM_8cond.npy")
dur_dist = np.load(predictor_DM_path+"duration_DM_8cond.npy")
predictors = np.stack((num_dist, rate_dist, dur_dist), axis=1)
std_predictors = StandardScaler().fit_transform(predictors) 

# run multiple linear regression on neural matrices 
for s in subjects:
    neural_DMs_ = np.load(neural_DMs_path+s+"_neural_DMs_pearson_125Hz_realigned_8cond")
    neural_DMs = np.mean(neural_DMs_, axis=0)
    betas = np.zeros((neural_DMs.shape[0], std_predictors.shape[1]))
    intercept = np.zeros(neural_DMs.shape[0])
    for tp in range(neural_DMs.shape[0]):
        std_neural_dist = StandardScaler().fit_transform(neural_DMs[tp].reshape(-1,1))
        std_neural_dist = np.squeeze(std_neural_dist, axis=1)
        LR.fit(std_predictors,std_neural_dist)
        betas[tp] = LR.coef_
        intercept[tp] = LR.intercept_
    np.save(save_path+"/betas/"+s+"_betas_125Hz_realigned_8cond", arr=betas)
    np.save(save_path+"/intercept/"+s+"_intercept_125Hz_realigned_8cond", arr=intercept)

