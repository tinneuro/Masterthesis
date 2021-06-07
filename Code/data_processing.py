import numpy as np
from numpy import save,load
from pathlib import Path

def load_stats_data(path):
    """Loading the data matrices"""
    percUP_matrix = load(Path.joinpath(path, "percUP_matrix.npy"))
    percDOWN_matrix = load(Path.joinpath(path, "percDOWN_matrix.npy"))

    meanUP_matrix = load(Path.joinpath(path, "meanUP_matrix.npy"))
    meanDOWN_matrix = load(Path.joinpath(path, "meanDOWN_matrix.npy"))

    cvUP_matrix = load(Path.joinpath(path, "cvUP_matrix.npy"))
    cvDOWN_matrix = load(Path.joinpath(path, "cvDOWN_matrix.npy"))

    scc_lag0_matrix = load(Path.joinpath(path, "scc_lag0_matrix.npy"))
    scc_lag1_matrix = load(Path.joinpath(path, "scc_lag1_matrix.npy"))

    #skewUP_matrix = load(Path.joinpath(path, "skewUP_matrix.npy")) # not used right now
    #skewDOWN_matrix = load(Path.joinpath(path, "skewDOWN_matrix.npy")) # not used right now

    stats_data = (percUP_matrix,percDOWN_matrix,meanUP_matrix,meanDOWN_matrix,cvUP_matrix,cvDOWN_matrix,scc_lag0_matrix,scc_lag1_matrix)

    return stats_data


def getnotnan_scc_data(data,regime_matrix,class_label):
    """Only return those gridpoints of each statistic where the SCC was computed.
    This means that we had at least 400 U,D transitions.
    Create class label for noise class."""
    percUP_matrix,percDOWN_matrix,meanUP_matrix,meanDOWN_matrix,cvUP_matrix,cvDOWN_matrix,scc_lag0_matrix,scc_lag1_matrix = data
    nan_scclag1 = np.isnan(scc_lag1_matrix)

    percup_notnan = percUP_matrix[~nan_scclag1]
    percdown_notnan = percDOWN_matrix[~nan_scclag1]
    meanup_notnan = meanUP_matrix[~nan_scclag1]
    meandown_notnan = meanDOWN_matrix[~nan_scclag1]
    cvup_notnan = cvUP_matrix[~nan_scclag1]
    cvdown_notnan = cvDOWN_matrix[~nan_scclag1]
    scclag0_notnan = scc_lag0_matrix[~nan_scclag1]
    scclag1_notnan = scc_lag1_matrix[~nan_scclag1]

    regime_notnan = regime_matrix[~nan_scclag1]

    if class_label == 0:
        label_notnan = np.zeros(len(percup_notnan))
    else:
        label_notnan = np.ones(len(percup_notnan))

    data_notnan = percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan

    return data_notnan,regime_notnan,label_notnan

def get_ratio_data(data_notnan):
    """Get the ratio statistics from all datapoints where transitions present.
    SCCratio, CVratio, Meanratio"""
    percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan = data_notnan
    scc_ratio = scclag1_notnan/scclag0_notnan
    cv_ratio = cvup_notnan/cvdown_notnan
    mean_ratio = meanup_notnan/meandown_notnan

    ratio_data = scc_ratio,cv_ratio,mean_ratio

    return ratio_data

def get_difference_data(data_notnan):
    """Obtain difference statistics from all datapoints where transitions present.
    SCCdiff,CVdiff,meandiff"""
    percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan = data_notnan
    scc_diff = scclag1_notnan-scclag0_notnan
    cv_diff = cvup_notnan-cvdown_notnan
    mean_diff = meanup_notnan-meandown_notnan

    diff_data = scc_diff,cv_diff,mean_diff

    return diff_data



def filtered_data(data,regime_matrix,class_label,filter_vars):
    """Filter the data by some specific condition. For example SCC > 0.1.
    This condition needs to be evaluated beforehand and the resulting matrix filter_vars with TRUE and FALSE is passed to this function."""
    percUP_matrix,percDOWN_matrix,meanUP_matrix,meanDOWN_matrix,cvUP_matrix,cvDOWN_matrix,scc_lag0_matrix,scc_lag1_matrix = data


    percup_filter = percUP_matrix[filter_vars]
    percdown_filter = percDOWN_matrix[filter_vars]
    meanup_filter = meanUP_matrix[filter_vars]
    meandown_filter = meanDOWN_matrix[filter_vars]
    cvup_filter = cvUP_matrix[filter_vars]
    cvdown_filter = cvDOWN_matrix[filter_vars]
    scclag0_filter = scc_lag0_matrix[filter_vars]
    scclag1_filter = scc_lag1_matrix[filter_vars]

    regime_filter = regime_matrix[filter_vars]

    nan_scclag1_filter = np.isnan(scclag1_filter)
    percup_filter = percup_filter[~nan_scclag1_filter]
    percdown_filter = percdown_filter[~nan_scclag1_filter]
    meanup_filter = meanup_filter[~nan_scclag1_filter]
    meandown_filter = meandown_filter[~nan_scclag1_filter]
    cvup_filter = cvup_filter[~nan_scclag1_filter]
    cvdown_filter = cvdown_filter[~nan_scclag1_filter]
    scclag0_filter = scclag0_filter[~nan_scclag1_filter]
    scclag1_filter = scclag1_filter[~nan_scclag1_filter]

    regime_filter = regime_filter[~nan_scclag1_filter]

    if class_label == 0:
        label_filter = np.zeros(len(percup_filter))
    else:
        label_filter = np.ones(len(percup_filter))

    data_filtered = percup_filter,percdown_filter,meanup_filter,meandown_filter,cvup_filter,cvdown_filter,scclag0_filter,scclag1_filter

    return data_filtered,regime_filter,label_filter

def get_data_array(data,ratio_data,w_data,regime_data):
    """Construct data matrix with n_datapoints x features using ratio statistics and regimes and w_matrix"""
    percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan = data
    scc_ratio,cv_ratio,mean_ratio = ratio_data

    X = np.array([percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag1_notnan,scclag0_notnan,scc_ratio,cv_ratio,mean_ratio,w_data,regime_data]).T

    return X

def get_data_array_reduced(data,ratio_data,w_data,regime_data):
    """Construct data matrix with n_datapoints x features using ratio statistics without scc_ratio but with regimes and w_matrix"""
    percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan = data
    scc_ratio,cv_ratio,mean_ratio = ratio_data

    X = np.array([percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag1_notnan,scclag0_notnan,cv_ratio,mean_ratio,w_data,regime_data]).T

    return X

def get_data_array_no_w_and_regime(data,ratio_data,w_data,regime_data):
    """Construct data matrix with  n_datapoints x features without regime and w_matrix"""
    percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan = data
    scc_ratio,cv_ratio,mean_ratio = ratio_data

    X = np.array([percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag1_notnan,scclag0_notnan,scc_ratio,cv_ratio,mean_ratio]).T

    return X

def get_400_transitions_data(data_path,regime_matrix,noise_type):
    """Create all data needed for classification on datapoints with >= 400 U,D transitions.
    noise_type: Mult = 0, Add = 1"""
    stats_data = load_stats_data(data_path)
    data_notnan,regime_notnan,noise_notnan = getnotnan_scc_data(stats_data,regime_matrix,noise_type)
    diff_data = get_difference_data(data_notnan)
    ratio_data = get_ratio_data(data_notnan)

    total_data = (data_notnan,diff_data,ratio_data,regime_notnan,noise_notnan)

    return total_data

def get_significant_scc_data(data_path,regime_matrix,noise_type):
    """Create all data needed for classification filtered for significant SCCDU and SCCUD values (>0.1)"""
    stats_data = load_stats_data(data_path)
    filter_stats =  np.logical_and(stats_data[-1] > 0.1,stats_data[-2] > 0.1)
    data_filter,regime_filter,noise_filter = filtered_data(stats_data,regime_matrix,noise_type,filter_stats)
    diff_data = get_difference_data(data_filter)
    ratio_data = get_ratio_data(data_filter)

    total_data = (data_filter,diff_data,ratio_data,regime_filter,noise_filter)

    return total_data

def data_with_scc_classify_noise_or_regime(data,noise_strength):
    """ Prepare data for classification into noise (or regime).
        Features which are used: percup_notnan,cv_diff,scc_diff
        We add the noise strength to the features for visualisation but not for training.

        want to classify into noise as first step, so we do not have the regime as information!
        also state what kind of noise strength the dataset has to check the distribution of errors across different noise strengths
        noise_strength = 0,1,2 for mult low,middle,high 3,4,5 for add low,middle,high
        careful! Small N -> High noise, Sigma low -> Low Noise"""

    data_notnan,diff_data,ratio_data,regime_notnan,noise_notnan = data

    percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan = data_notnan
    scc_ratio,cv_ratio,mean_ratio = ratio_data
    scc_diff,cv_diff,mean_diff = diff_data

    noise_strength_notnan = noise_strength*np.ones(len(percup_notnan))

    X = np.array([percup_notnan,cv_diff,scc_diff,noise_strength_notnan]).T
    labels_noise = noise_notnan
    labels_regime = regime_notnan

    return X,labels_noise,labels_regime

def data_with_all_features_classify_noise_or_regime(data,noise_strength):
    """Function as  data_with_scc_classify_noise_or_regime but use all features.
    Features: percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan,mean_diff,cv_diff,scc_diff,noise_strength_notnan"""
    data_notnan,diff_data,ratio_data,regime_notnan,noise_notnan = data

    percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan = data_notnan
    scc_ratio,cv_ratio,mean_ratio = ratio_data
    scc_diff,cv_diff,mean_diff = diff_data

    noise_strength_notnan = noise_strength*np.ones(len(percup_notnan))
    # exclude the ratio_data!
    X = np.array([percup_notnan,percdown_notnan,meanup_notnan,meandown_notnan,cvup_notnan,cvdown_notnan,scclag0_notnan,scclag1_notnan,mean_diff,cv_diff,scc_diff,noise_strength_notnan]).T
    labels_noise = noise_notnan
    labels_regime = regime_notnan

    return X,labels_noise,labels_regime
