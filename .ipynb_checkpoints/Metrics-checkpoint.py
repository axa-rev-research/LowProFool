import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.neighbors import NearestNeighbors


def get_metrics(config, list_metrics, n_neighbors=2):
    """
    Generates an adversarial examples x' from an original sample x. Expected to contain
            Dataset, MaxIters, Alpha, Lambda, TrainData, TestData, ValidData, Scaler,
            FeatureNames, Target, Weights, Bounds, Model
    :param config: dictionnary containing the configuration for the experiment
    :param list_metrics: dictionnary containing the metrics to be computed. Choose from
            from SuccessRate, iter_means, iter_std, normdelta_median, normdelta_mean,
            n_std, weighted_median, weighted_mean, w_std, mean_dists_at_org,
            median_dists_at_org, mean_dists_at_tgt, mean_dists_at_org_weighted, mdow_std,
            median_dists_at_org_weighted, mean_dists_at_tgt_weighted, mdtw_std, prop_same_class_arg_org,
            prop_same_class_arg_adv,
    :param n_neighbors: number of neighbors to compute the distance to n_neighbors closest neighbors
    """

    metrics_for_conf = []
    df_test = config['TestData']
    dfs_adv = config['AdvData']
    
    for method, df_adv in dfs_adv.items():
        metrics_for_method = [method]
        # Get success rate before removing samples from dataframe
        if list_metrics['SuccessRate']:
            sr = metric_success_rate_for(df_adv)
            metrics_for_method.append(sr)

        # Removes samples that did cross frontier
        df_adv = remove_non_converted(df_adv)        
        df_adv = add_normdelta_to(df_adv, config, df_test)

        # Adding proportion of neighbors from diff classes
        df_adv, df_adv_weighted = add_maj_neighbors(df_adv, df_test, config, n_neighbors=n_neighbors)            

        # Mean, std, number of iterations
        if list_metrics['iter_means']:
            means_iters, stds_iters = mean_norm_for_col(df_adv, col='iters')
            metrics_for_method.append(means_iters)
            if list_metrics['iter_std']:
                metrics_for_method.append(stds_iters)

        # Median, norm of perturbation
        if list_metrics['normdelta_median']:
            median = median_norm_for_col(df_adv, col='normdelta')
            metrics_for_method.append(median)

        # Mean, std, norm of perturbation
        if list_metrics['normdelta_mean']:
            means, stds = mean_norm_for_col(df_adv, col='normdelta')
            metrics_for_method.append(means)
            if list_metrics['n_std']:
                metrics_for_method.append(stds)

        # Median, norm of perturbation, weighted
        if list_metrics['weighted_median']:
            median_w = median_norm_for_col(df_adv, col='normdelta_weighted')
            metrics_for_method.append(median_w)

        # Mean, std, norm of perturbation, weighted
        if list_metrics['weighted_mean']:
            means_w, stds_w = mean_norm_for_col(df_adv, col='normdelta_weighted')
            metrics_for_method.append(means_w)
            if list_metrics['w_std']:
                metrics_for_method.append(stds_w) 

        # Mean, std, number of neighbors of a particular class at perturbed sample
        if list_metrics['mean_dists_at_org']:
            mean, std = mean_norm_for_col(df_adv, col='mean_dists_at_org')
            metrics_for_method.append(mean)

        # Mean, std, number of neighbors of a particular class at perturbed sample
        if list_metrics['median_dists_at_org']:
            med = median_norm_for_col(df_adv, col='mean_dists_at_org')
            metrics_for_method.append(med)

        # Mean, std, number of neighbors of a particular class at perturbed sample
        if list_metrics['mean_dists_at_tgt']:
            mean, std = mean_norm_for_col(df_adv, col='mean_dists_at_tgt')
            metrics_for_method.append(mean)

        # Mean, std, number of neighbors of a particular class at perturbed sample
        if list_metrics['mean_dists_at_org_weighted']:
            mean, std = mean_norm_for_col(df_adv_weighted, col='mean_dists_at_org')
            metrics_for_method.append(mean)
            if list_metrics['mdow_std']:
                metrics_for_method.append(std)

        # Mean, std, number of neighbors of a particular class at perturbed sample
        if list_metrics['median_dists_at_org_weighted']:
            median = median_norm_for_col(df_adv_weighted, col='mean_dists_at_org')
            metrics_for_method.append(median)

        # Mean, std, number of neighbors of a particular class at perturbed sample
        if list_metrics['mean_dists_at_tgt_weighted']:
            mean, std = mean_norm_for_col(df_adv_weighted, col='mean_dists_at_tgt')
            metrics_for_method.append(mean)
            if list_metrics['mdtw_std']:
                metrics_for_method.append(std)

        # Mean, std, number of neighbors of a particular class at perturbed sample
        if list_metrics['prop_same_class_arg_org']:
            mean, std = mean_norm_for_col(df_adv, col='prop_same_class_arg_org')
            metrics_for_method.append(mean)

        # Mean, std, number of neighbors of a particular class at perturbed sample
        if list_metrics['prop_same_class_arg_adv']:
            mean, std = mean_norm_for_col(df_adv, col='prop_same_class_arg_adv')
            metrics_for_method.append(mean)
            
        metrics_for_conf.append(metrics_for_method)
    return metrics_for_conf
  
def metric_success_rate_for(df):
    return len(df[df['orig_pred'] != df['adv_pred']]) / df.shape[0]

def remove_non_converted(df):
    df_return = df.copy()
    return df[df['orig_pred'] != df['adv_pred']]

def mean_norm_for_col(df, col):
    tmp = df[col]    
    mean, std = np.mean(tmp), np.std(tmp)
    return (mean, std)

def median_norm_for_col(df, col):
    tmp = df[col]    
    median = np.median(tmp)
    return median

def add_normdelta_to(df_adv, conf, df):
    # Drop columns if already there
    df_return = df_adv.copy()
    if 'normdelta' in df_return.columns:
        df_return = df_return.drop(columns='normdelta')
    if 'normdelta_weighted' in df_return.columns:
        df_return = df_return.drop(columns='normdelta_weighted')
        
    feature_names = conf['FeatureNames']
    weights = conf['Weights']

    norms = []
    norms_weighted = []
    
    # Iterate over all rows
    for index, row in df_return.iterrows():
        orig = df.loc[index][feature_names].values
        adv = row[feature_names].values 
        
        # Compute deltas
        delta = np.abs(orig-adv)
        assert(len(delta) == len(weights))
        
        # Norms delta
        norms.append(np.linalg.norm(delta))
        
        # Norms delta weighted
        norms_weighted.append(np.linalg.norm(delta * weights))

    df_return.insert(0, 'normdelta', norms)
    df_return.insert(0, 'normdelta_weighted', norms_weighted)
    
    return df_return


def get_majority_neighbors(df_adv, df_orig, conf, knn, n_neighbors):
    
    # orig, adv
    mean_dists = [[], []]
    prop_same_class = [[], []]
    
    feature_names = conf['FeatureNames']
    target = conf['Target']    
    
    # For each sample
    for index, row in df_adv.iterrows():
        
        orig = df_orig.loc[index][feature_names].values
        adv = row[feature_names].values
        
        preds = [row['orig_pred'], row['adv_pred']]
        samples = [orig, adv]
        
        for i in range(len(preds)):
            
            sample = samples[i]
            pred = preds[i]
            
            distance, neighbors_idxs = knn.kneighbors([sample], n_neighbors)
            neighbors_samples = df_orig.iloc[neighbors_idxs[0]]
            
           
            distance = [distance[0][1:]]
            neighbors_idxs = [neighbors_idxs[0][1:]]
            

            # Distance to closest neighbors
            if len(distance[0]) > 0 :
                dst_mean = np.mean(distance[0])
            else:
                print('Error, no neighbor found')
            mean_dists[i].append(dst_mean)
            
            neighbors_pts_target = np.array(neighbors_samples[target]).astype(int)
            prop = list(neighbors_pts_target).count(pred)
            prop_same_class[i].append(float(prop)/float(n_neighbors))
                        
    return mean_dists, prop_same_class

def add_maj_neighbors_to(df_adv, df_orig, conf, knn, n_neighbors):
    df_return = df_adv.copy()
    
    if 'mean_dists_at_org' in df_return.columns:
        df_return = df_return.drop(columns='mean_dists_at_org')
    if 'mean_dists_at_tgt' in df_return.columns:
        df_return = df_return.drop(columns='mean_dists_at_tgt')
    if 'prop_same_class_arg_org' in df_return.columns:
        df_return = df_return.drop(columns='prop_same_class_arg_org')
    if 'prop_same_class_arg_adv' in df_return.columns:
        df_return = df_return.drop(columns='prop_same_class_arg_adv')
        
    mean_dists, prop_same_class = get_majority_neighbors(df_adv, df_orig, conf, knn, n_neighbors)
    
    df_return.insert(0, 'mean_dists_at_org', mean_dists[0])
    df_return.insert(0, 'mean_dists_at_tgt', mean_dists[1])

    df_return.insert(0, 'prop_same_class_arg_org', prop_same_class[0])
    df_return.insert(0, 'prop_same_class_arg_adv', prop_same_class[1])
    
    return df_return

def scale_data(conf, df_orig):
    print('Before')
    print(df.describe(include='all'))
    print(weights)
    for col, weight in zip(list(df.columns), weights):
        df[col] = df[col].apply(lambda x: x * weight)
        
    bounds = [[bounds[i][x] * weight for x, weight in enumerate(weights)] for i in range(len(bounds))]
    print(df.describe(include='all'))
    return df, bounds

def weighted_distance(x, y, w):
    sum_ = 0
    assert(len(x) == len(y) == len(w))
    for i in range(len(x)):
        sum_ += (w[i] * (y[i] - x[i])) ** 2
    sum_ = np.sqrt(sum_)
    return sum_

def add_maj_neighbors(df_adv, df_orig, conf, n_neighbors):
    # Otherwise we have issues because the KNN returns indexes in len(df) and not based on the real indexes on the samples
    df_adv = df_adv.reset_index().drop(columns=['index'])
    df_orig = df_orig.reset_index().drop(columns=['index'])
    weights = conf['Weights']

    assert(weights[0] > 0)

    feature_names = conf['FeatureNames']
    target = conf['Target']
        
        
    knn = NearestNeighbors(n_neighbors, metric='l2')
    knn.fit(df_orig[feature_names], df_orig[target])
    
    knn_weighted = NearestNeighbors(n_neighbors, metric=weighted_distance, metric_params={'w' : weights})
    knn_weighted.fit(df_orig[feature_names], df_orig[target])

    
    df_adv_return = add_maj_neighbors_to(df_adv, df_orig, conf, knn, n_neighbors)
    df_adv_weighted = add_maj_neighbors_to(df_adv, df_orig, conf, knn_weighted, n_neighbors)
    
    return df_adv_return, df_adv_weighted