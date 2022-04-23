from typing import NamedTuple
from kfp.v2.dsl import (
    Artifact, 
    Dataset, 
    Input,
    InputPath, 
    Model, 
    Output,
    OutputPath,
    Metrics,
    HTML,
    component
)

def printing(
    msg: str,
):
    print(msg)    


def data_quality_check(
    pred_month: str,
    dataset: Input[Dataset],
) -> NamedTuple(
    "Outputs",
    [
        ("qc_status", bool),  # Return generic Artifact.
    ],
):
    import pandas as pd

    df = pd.read_parquet(dataset.path)

    # L1 QC
    print(df.head())
    print("Dataset shape: ", df.shape)
    print("To find null in dataset before pre-processing:\n", df.isnull().sum())

    qc_status = True
    return (
        qc_status,
    )

# True means passes, False means failed
def kld_drift_check(
    dataset_p: Input[Dataset], # current
    dataset_q: Input[Dataset], # previous
    features_dict_str: str,
    drift_threshold: float,
    mode: str,
    feature_drifts: Output[Metrics]
) -> NamedTuple(
    "Outputs",
    [
        ("drift_status", bool),
    ],
):
    import pandas as pd
    from math import log2
    import json
    dataset_p_df = pd.read_parquet(dataset_p.path)
    dataset_q_df = pd.read_parquet(dataset_q.path)
    
    if len(dataset_q_df) > 0:
        features_dict = json.loads(features_dict_str)

        kld_scores_dict = {}
        global_drift = 0
        for col, bucket in features_dict.items():
            series_p = dataset_p_df[col] 
            series_q = dataset_q_df[col] 

            if bucket:
                bucket.insert(0, -float("inf"))
                bucket.append(float("inf"))
                bucket = [float(i) for i in bucket]
                series_p = pd.cut(series_p, bucket)
                series_q = pd.cut(series_q, bucket)

            series_p_density = series_p.value_counts(normalize=True).rename("density_p")
            series_p_density.index.rename("bins", inplace=True)
            series_q_density = series_q.value_counts(normalize=True).rename("density_q")
            series_q_density.index.rename("bins", inplace=True)

            df_pq = pd.concat([series_p_density, series_q_density], axis=1)

            zero_approximation = 1e-6
            df_pq.fillna(zero_approximation, inplace=True)
            df_pq.replace(0, zero_approximation, inplace=True)

            kld = df_pq.apply(lambda x: x.density_p * log2(x.density_p/x.density_q), axis=1).sum()
            kld = round(kld, 3)
            feature_drifts.log_metric(f"{mode}_feature_drift_{col}", kld)
            global_drift+=kld
        feature_drifts.log_metric(f"{mode}_global_drift", global_drift)

        if global_drift > drift_threshold:
            drift_status = False
        else:
            drift_status = True

    else:
        print('No dataset q, possibility of cold start, passing through.')
        drift_status = False

    return (
        drift_status,
    )

# True means passes, False means failed
def hd_drift_check(
    dataset_p: Input[Dataset], # current
    dataset_q: Input[Dataset], # previous
    features_dict_str: str,
    feature_importance_dict_str: str,
    drift_threshold: float,
    mode: str,
    feature_drifts: Output[Metrics]
) -> NamedTuple(
    "Outputs",
    [
        ("drift_status", bool),
    ],
):
    import pandas as pd
    from math import log2
    import json
    import numpy as np
    from numpy import rec
    dataset_p_df = pd.read_parquet(dataset_p.path)
    dataset_q_df = pd.read_parquet(dataset_q.path)

    if len(dataset_q_df) > 0:
        features_dict = json.loads(features_dict_str)

        #Get feature importance vector
        feature_importance_dict = json.loads(feature_importance_dict_str)
        if not feature_importance_dict: #this means all features are considered important by the data scientist
            feature_importance_dict = {key: 1.0 for key, val in features_dict.items()}
        else:
            assert(sum(list(feature_importance_dict.values())) == 1.0), 'Boosted value of feature importance dict must be equal to 1'
        feature_importance_dict_keys = list(feature_importance_dict.keys())
        all_features_dict_keys = list(features_dict.keys())



        #assign null weights to features not being used for drift check
        feature_importance_dict_new = {}
        if(len(all_features_dict_keys)!=len(feature_importance_dict_keys)):
            for all_key in all_features_dict_keys:
                if all_key not in feature_importance_dict_keys:
                    feature_importance_dict_new[all_key] = 0
                else:
                    feature_importance_dict_new[all_key] = feature_importance_dict[all_key]
            feature_importance_dict = feature_importance_dict_new

        global_drift = 0
        for col, bucket in features_dict.items():
            try:
                zero_approximation = 1e-6
                series_p = dataset_p_df[col]
                series_p.fillna(zero_approximation, inplace=True)

                series_q = dataset_q_df[col]
                series_q.fillna(zero_approximation, inplace=True)

                p = 0
                q = 0
                if bucket: #this means if a bin range has been defined already by the data scientist

                    bin_p = pd.cut(series_p,bins=bucket).value_counts(dropna=True,sort=False)
                    bin_q = pd.cut(series_q,bins=bucket).value_counts(dropna=True,sort=False)

                    #Compute the probabilities for the bins
                    p = bin_p.apply(lambda x: x/bin_p.sum()).to_numpy()
                    q = bin_q.apply(lambda x: x/bin_q.sum()).to_numpy()

                    #replace nan values in probabilities with 0
                    p = np.nan_to_num(p)
                    q = np.nan_to_num(q)
                else: #no bucket hence perform auto binning for categorical data drift computation
                    #perform the automated binning since thie feature is categorical
                    # Get a tuple of unique values & their frequency in numpy array
                    arr1 = series_p
                    arr2 = series_q

                    arr1_length = len(arr1)
                    uniqueValues1, occurCount1 = np.unique(arr1, return_counts=True)
                    arr1_prob = np.divide(occurCount1,arr1_length)

                    # Get a tuple of unique values & their frequency in numpy array
                    arr2_length = len(arr2)
                    uniqueValues2, occurCount2 = np.unique(arr2, return_counts=True)
                    arr2_prob = np.divide(occurCount2,arr2_length)

                    # yields the elements in `arr2` that are NOT in `arr1`
                    elements_to_add_in_arr1 = np.setdiff1d(arr2,arr1)

                    # yields the elements in `arr1` that are NOT in `arr2`
                    elements_to_add_in_arr2 = np.setdiff1d(arr1,arr2)

                    uniqueValues1 = np.concatenate([uniqueValues1,elements_to_add_in_arr1])
                    arr1_prob = np.concatenate([arr1_prob,np.zeros(len(elements_to_add_in_arr1))])
                    c = rec.fromarrays([uniqueValues1, arr1_prob])
                    c.sort()
                    arr1_prob = c.f1

                    uniqueValues2 = np.concatenate([uniqueValues2,elements_to_add_in_arr2])
                    arr2_prob = np.concatenate([arr2_prob,np.zeros(len(elements_to_add_in_arr2))])
                    c = rec.fromarrays([uniqueValues2, arr2_prob])
                    c.sort()
                    arr2_prob = c.f1

                    p = arr1_prob
                    q = arr2_prob

                hld = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
                hld = round(hld, 3)
                boosted_hld = feature_importance_dict[col] * hld
                feature_drifts.log_metric(f"{mode}_feature_drift_{col}", boosted_hld)
                global_drift += (boosted_hld)
            except:
                pass
        feature_drifts.log_metric(f"{mode}_global_drift", global_drift)

        if global_drift > drift_threshold:
            drift_status = False
        else:
            drift_status = True

    else:
        print('No dataset q, possibility of cold start, force model training...')
        drift_status = False

    return (
        drift_status,
    )

def get_data_check_result(
    data_quality_check: bool,
    data_drift_check: bool,
    mode: str,
) -> NamedTuple(
    "Outputs",
    [
        ('retrain_model', bool),
        ("alert_msg", str)
    ],
):

    retrain_model = False
    alert_msg = ''
    all_checks = {
        f'{mode}_quality_check': data_quality_check,
        f'{mode}_drift_check': data_drift_check,
    }

    if all(all_checks.values()):
        print('All data integrity checks passed.')

    else:
        failed_checks = ', '.join([key for key, check in all_checks.items() if not check])
        
        retrain_model = True
        alert_msg = f'Data integrity check(s) failed : {failed_checks}'

    return (
        retrain_model,
        alert_msg,
    )
