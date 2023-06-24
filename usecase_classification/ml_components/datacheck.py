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

def grand_drift_check(
    bucket_name: str,
    src_run_dt: str,
    project_name: str,
    dataset_p: Input[Dataset], # current
    dataset_q: Input[Dataset], # previous
    feature_importance_dict_str: str,
    numerical_drift_partition_threshold: float,
    numerical_importance_partition_threshold: float,
    categorical_drift_partition_threshold: float,
    categorical_importance_partition_threshold: float,
    category_threshold: int,
    delta: int,
    mode: str,
    data_drift_view: Output[HTML],
) -> NamedTuple(
    "Outputs",
    [
        ("dq_status", bool),
        ("drift_status", bool),
        ("num_quadrant_dict", str),
        ("cat_quadrant_dict", str),
        ("feature_drift_dict", str),
    ],
):
    from scipy.stats import wasserstein_distance
    from math import log2

    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    import json
    import copy
    import time

    import base64
    import io

    # Start capturing graphs using io
    def plot_to_html(tmpfile):
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        plot_as_html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        return plot_as_html

    def PlotSortedVisualizations(feature_drift_info):
        feature_drift_info = sorted(feature_drift_info, key = lambda x: (x[0]), reverse=True)
        feature_graphs = []

        for i in range(len(feature_drift_info)):
            plt.figure()
            if(feature_drift_info[i][6]=="c"): # categorical feature
                hld, X_axis, p, q, arr_1_order, col = feature_drift_info[i][0], feature_drift_info[i][1], feature_drift_info[i][2], feature_drift_info[i][3], feature_drift_info[i][4], feature_drift_info[i][5]

                plt.bar(X_axis - 0.2, p, 0.4, label = 'P')
                plt.bar(X_axis + 0.2, q, 0.4, label = 'Q')
                plt.xticks(X_axis, arr_1_order, rotation='vertical')
                plt.title("feature: "+str(col)+" and drift: "+str(hld*100.0)+"%")
                plt.xlabel("Category")
                plt.ylabel("Probability")
                plt.legend()
            else:
                emd, feature_a, feature_b, col = feature_drift_info[i][0], feature_drift_info[i][2], feature_drift_info[i][3], feature_drift_info[i][5]

                feature_a_copy = copy.deepcopy(feature_a)
                feature_b_copy = copy.deepcopy(feature_b)

                c = np.append(feature_a_copy, feature_b_copy)

                temp_c = np.unique(c)
                global_min, global_max = temp_c[int(0.03*len(temp_c))], temp_c[int(0.97*len(temp_c))] # eliminate the outliers

                feature_a_copy = feature_a_copy[ (feature_a_copy >= global_min) & (feature_a_copy <= global_max) ]
                feature_b_copy = feature_b_copy[ (feature_b_copy >= global_min) & (feature_b_copy <= global_max) ]


                plt.title("feature: "+str(col)+" and drift: "+str(emd*100.0)+"%")
                plt.hist(feature_a_copy, density=False, bins=100, fc=(0, 0, 1, 0.3), edgecolor='black', linewidth=0.5, label = 'P')
                plt.hist(feature_b_copy, density=False, bins=100, fc=(1, 0, 0, 0.3), edgecolor='black', linewidth=0.5, label = 'Q')
                plt.xlabel("Values")
                plt.ylabel("Count")
                plt.legend()

            feature_graphs.append(plot_to_html(io.BytesIO()))
        return feature_graphs

    def CategoricalAutoBinning(feature_a, feature_b, col):

        uniqueValues1 = np.asarray(feature_a.value_counts().keys())
        uniqueValues2 = np.asarray(feature_b.value_counts().keys())

        arr1_prob = np.asarray(feature_a.value_counts(normalize=True).tolist())
        arr2_prob = np.asarray(feature_b.value_counts(normalize=True).tolist())

        arr1, arr2 = feature_a.to_numpy(), feature_b.to_numpy()

        if np.array_equal(uniqueValues1, uniqueValues2):
            p = arr1_prob
            q = arr2_prob

            hld = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
            hld = round(hld, 3)

            X_axis = np.arange(len(uniqueValues1))

            return hld, X_axis, p, q, uniqueValues1, "c"
        else:
            # yields the elements in `arr2` that are NOT in `arr1`
            elements_to_add_in_arr1 = np.setdiff1d(arr2, arr1)

            # yields the elements in `arr1` that are NOT in `arr2`
            elements_to_add_in_arr2 = np.setdiff1d(arr1, arr2)

            uniqueValues1 = np.concatenate([uniqueValues1, elements_to_add_in_arr1])
            arr1_prob = np.concatenate([arr1_prob, np.zeros(len(elements_to_add_in_arr1))])
            c = np.rec.fromarrays([uniqueValues1, arr1_prob])
            c.sort()
            arr1_prob = c.f1
            arr_1_order = c.f0

            uniqueValues2 = np.concatenate([uniqueValues2, elements_to_add_in_arr2])
            arr2_prob = np.concatenate([arr2_prob, np.zeros(len(elements_to_add_in_arr2))])
            c = np.rec.fromarrays([uniqueValues2, arr2_prob])
            c.sort()
            arr2_prob = c.f1
            arr_2_order = c.f0

            p = arr1_prob
            q = arr2_prob

            hld = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
            hld = round(hld, 6)

            X_axis = np.arange(len(arr_1_order))

            return hld, X_axis, p, q, arr_1_order, "c"

    def NormalizedEMDScore(a, b):
        if np.min(b)>np.max(a) or np.min(a)>np.max(b):
            return 1.0

        c = np.append(a, b)
        temp_c = np.unique(c)
        global_min, global_max = temp_c[int(0.03*len(temp_c))], temp_c[int(0.97*len(temp_c))] #eliminate the outliers

        emd_score_numerical = 0
        if global_min != global_max:
            norm_a = (a - global_min)/(global_max-global_min)
            norm_b = (b - global_min)/(global_max-global_min)

            norm_a = norm_a[(norm_a >= 0) & (norm_a <= 1) ]
            norm_b = norm_b[(norm_b >= 0) & (norm_b <= 1) ]

            emd_score_numerical = wasserstein_distance(norm_a, norm_b)
            emd_score_numerical = round(emd_score_numerical, 6)

        return emd_score_numerical

    def GetQuadrantInfo(
        drift_score,
        drift_partition_threshold,
        feature_importance,
        importance_partition_threshold
    ):
        if drift_score>=drift_partition_threshold and feature_importance>=importance_partition_threshold: #Quadrant 1
            return "Quadrant 1"
        elif drift_score>=drift_partition_threshold and feature_importance<importance_partition_threshold: #Quadrant 2
            return "Quadrant 2"
        elif drift_score<drift_partition_threshold and feature_importance<importance_partition_threshold: #Quadrant 3
            return "Quadrant 3"
        elif drift_score<drift_partition_threshold and feature_importance>=importance_partition_threshold: #Quadrant 4
            return "Quadrant 4"

    def PlotQuadrant(
        Drift_scores,
        importance,
        feature_names,
        drift_partition_threshold,
        importance_partition_threshold,
        f_type
    ):
        # Plot drift quadrant
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.rcParams.update({'font.size': 12})

        plt.figure()
        plt.scatter(importance, Drift_scores)
        plt.xlim(-0.01, 1)
        plt.ylim(-0.01, 1)

        # Plot partition lines  
        plt.axhline(y=drift_partition_threshold, color='k', linestyle='--', linewidth=1)           
        plt.axvline(x=importance_partition_threshold, color='k', linestyle='--', linewidth=1) 

        # Quadrant Marker          
        plt.text(x=(1+importance_partition_threshold)/2, y=(1+drift_partition_threshold)/2, s="Q1", fontsize=14, color='b')
        plt.text(x=importance_partition_threshold/2, y=(1+drift_partition_threshold)/2, s="Q2", fontsize=14, color='b')
        plt.text(x=importance_partition_threshold/2, y=(drift_partition_threshold)/2, s="Q3", fontsize=14, color='b')
        plt.text(x=(1+importance_partition_threshold)/2, y=(drift_partition_threshold)/2, s="Q4", fontsize=14, color='b') 

        # Feature names
        for i in range(len(importance)):
            plt.text(importance[i], y=Drift_scores[i], s=feature_names[i], alpha=0.8)

        plt.xlabel('Feature Importance')
        plt.ylabel('Drift Score')
        plt.title("Drift quadrant for "+str(f_type)+" features")

    def basicDQCheck(
        dataset_p_df,
        dataset_q_df,
        delta
    ):
        dq_check_status = True # initially, assume there are no DQ issues

        # Check difference in row count
        min_rows = min(dataset_p_df.shape[0], dataset_q_df.shape[0])
        max_rows = max(dataset_p_df.shape[0], dataset_q_df.shape[0])

        tolerance = 100 - ((min_rows/max_rows)*100.0)
        dq_check_fail_reason = []
        if tolerance > delta:
            dq_check_status = False
            dq_check_fail_reason.append("row count difference greater than 10%")

        for col in dataset_p_df.columns:
            if dataset_p_df[col].isna().sum() > 0:
                dq_check_status = False
                dq_check_fail_reason.append("Dataset P has null values")
                break

        for col in dataset_q_df.columns:
            if dataset_q_df[col].isna().sum() > 0:
                dq_check_status = False
                dq_check_fail_reason.append("Dataset Q has null values")
                break

        if len(dq_check_fail_reason) > 0:
            print("Reasons for DQ check fail: ")
            for reason in dq_check_fail_reason:
                print(reason)
        return dq_check_status

    dataset_p_df = pd.read_parquet(dataset_p.path)
    dataset_q_df = pd.read_parquet(dataset_q.path)

    # Initialization

    drift_status = True
    cat_drift_info, num_drift_info, general_drift_info, feature_graphs = [], [], [], []

    num_quadrant_dict = {f'Quadrant {i+1}': [] for i in range(0, 4, 1)}
    cat_quadrant_dict = {f'Quadrant {i+1}': [] for i in range(0, 4, 1)}

    global_drift, hld, emd = 0, 0, 0
    feature_drift_dict = {}
    feature_drift_info, Drift_scores, importance, feature_names = [], [], [], []

    total_numerical_imp, total_categorical_imp = 0, 0
    numerical_count, categorical_count = 0, 0
    numerical_total_drift, categorical_total_drift = 0, 0

    num_drift_scores, cat_drift_scores = [], []
    num_importance, cat_importance = [], []
    num_feature_names, cat_feature_names = [], []

    dq_status = basicDQCheck(dataset_p_df, dataset_q_df, delta)

    numerical_drift_threshold = numerical_drift_partition_threshold
    categorical_drift_threshold = categorical_drift_partition_threshold

    if len(dataset_q_df) > 0 and len(dataset_p_df) > 0:

        all_features_dict_keys = list(dataset_q_df.columns)
        # Get feature importance vector
        feature_importance_dict = json.loads(feature_importance_dict_str)
        if not feature_importance_dict: # this means all features are considered important by DS
            for col in dataset_q_df.columns:
                feature_importance_dict[col] = 1.0
        else:
            # normalize the feature importance here
            feature_imp = np.asarray(list(feature_importance_dict.values()))
            if np.max(feature_imp) > 1 : # only normalize if feature importances are not normalized by DS
                norm = np.linalg.norm(feature_imp)
                feature_imp = feature_imp/norm
                feature_imp_dict = {key: norm_val for (key, val), norm_val in zip(feature_importance_dict.items(), feature_imp.tolist())}
                feature_importance_dict.update(feature_imp_dict)

            no_boost_features = [i for i in all_features_dict_keys if i not in feature_importance_dict]
            if no_boost_features:
                for key in no_boost_features:
                    feature_importance_dict[key] = 0

        for col in dataset_q_df.columns:
            try:
                series_p = dataset_p_df[col]
                series_q = dataset_q_df[col]

                drift_score = 0
                boosted_hld, boosted_emd = 0, 0
                # Categorical data hence perform HLD with auto binning
                if (
                    (series_p.nunique() < category_threshold or series_q.nunique() < category_threshold) or \
                    (series_p.dtype == 'object' and series_q.dtype == 'object')
                ): 
                    hld, X_axis, p, q, arr_1_order, feature_type = CategoricalAutoBinning(
                        series_p,
                        series_q,
                        col
                    )
                    feature_drift_dict[col] = hld
                    feature_drift_info.append([hld, X_axis, p, q, arr_1_order, col, feature_type])
                    total_categorical_imp += feature_importance_dict[col]

                    categorical_total_drift += hld
                    categorical_count += 1
                    quadrant_classification = GetQuadrantInfo(
                        hld,
                        categorical_drift_partition_threshold,
                        feature_importance_dict[col],
                        categorical_importance_partition_threshold
                    )
                    cat_quadrant_dict[quadrant_classification].append(col)

                    cat_drift_scores.append(hld)
                    cat_importance.append(feature_importance_dict[col])
                    cat_feature_names.append(col)

                # Numerical data hence perform normalized EMD
                else:
                    tic_num = time.time()
                    feature_a, feature_b = series_p.to_numpy(), series_q.to_numpy()
                    emd = NormalizedEMDScore(feature_a, feature_b)
                    feature_drift_dict[col] = emd
                    feature_drift_info.append([emd, 0, feature_a, feature_b, 0, col, "n"])
                    total_numerical_imp += feature_importance_dict[col]

                    numerical_total_drift += emd
                    numerical_count += 1
                    quadrant_classification = GetQuadrantInfo(emd, numerical_drift_partition_threshold, feature_importance_dict[col], numerical_importance_partition_threshold)
                    num_quadrant_dict[quadrant_classification].append(col)

                    num_drift_scores.append(emd)
                    num_importance.append(feature_importance_dict[col])
                    num_feature_names.append(col)

            except:
                pass

        if categorical_count:
            # Plot categorical drift quadrant
            PlotQuadrant(
                cat_drift_scores,
                cat_importance,
                cat_feature_names,
                categorical_drift_partition_threshold,
                categorical_importance_partition_threshold,
                "categorical"
            )
            categorical_total_drift = categorical_total_drift/categorical_count
            cat_drift_info.append("Overall drift in categorical features: "+str(categorical_total_drift)+" = "+str(categorical_total_drift*100)+"%")
            cat_drift_info.append("Categorical features drift threshold input by DS: "+str(categorical_drift_threshold)+" = "+str(categorical_drift_threshold*100)+"%")
            cat_drift_info.append("Total importance of categorical features input by DS: "+str(total_categorical_imp))
            cat_drift_info.append("Categorical features quadrant dictionary: "+str(cat_quadrant_dict))
            cat_drift_info.append('<br>')
            cat_graph = plot_to_html(io.BytesIO())

        if numerical_count:
            # Plot numerical drift quadrant
            PlotQuadrant(
                num_drift_scores,
                num_importance,
                num_feature_names,
                numerical_drift_partition_threshold,
                numerical_importance_partition_threshold,
                "numerical"
            )
            numerical_total_drift = numerical_total_drift/numerical_count
            num_drift_info.append("Overall drift in numerical features: "+str(numerical_total_drift)+" = "+str(numerical_total_drift*100)+"%\n")
            num_drift_info.append("Numerical features drift threshold input by DS: "+str(numerical_drift_threshold)+" = "+str(numerical_drift_threshold*100)+"%")
            num_drift_info.append("Total importance of numerical features input by DS: "+str(total_numerical_imp))
            num_drift_info.append("Numerical features quadrant dictionary: "+str(num_quadrant_dict))
            num_drift_info.append('<br>')
            num_graph = plot_to_html(io.BytesIO())

        # If any of the following cases are satisfied, then the re-training is flagged
        retrain_reasons = []
        # case 1: If both thresholds are exceeded
        if((numerical_total_drift >= numerical_drift_threshold) and (categorical_total_drift >= categorical_drift_threshold)):
            retrain_reasons.append("Retrain - because both numerical and categorical drifts exceeded thresholds")
            drift_status = False

        # case 2: If numerical threshold is exceeded and numerical features are more important than categorical
        if((numerical_total_drift >= numerical_drift_threshold) and (total_numerical_imp>=total_categorical_imp)):
            retrain_reasons.append("Retrain - because numerical drift exceeded threshold and numerical features more/equally important than categorical features")
            drift_status = False

        # case 3: If categorical threshold is exceeded and categorical features are more important than numerical
        if((categorical_total_drift >= categorical_drift_threshold) and (total_categorical_imp>=total_numerical_imp)):    
            retrain_reasons.append("Retrain - because categorical drift exceeded threshold and categorical features more/equally important than numerical features")
            drift_status = False

        # case 4: If any feature falls under quadrant 1
        if(len(num_quadrant_dict["Quadrant 1"])>0 or len(cat_quadrant_dict["Quadrant 1"])>0):
            retrain_reasons.append("Retrain - because at least one feature falls in Q1 of the drift quadrant")
            drift_status = False

        if(len(retrain_reasons)>0):
            general_drift_info.append("Re-train for the following reasons: ")
            for reason in retrain_reasons:
                general_drift_info.append(reason)
            general_drift_info.append('<br>')

        general_drift_info.append("All Features drift dictionary: "+str(feature_drift_dict))
        general_drift_info.append('<br>')

        #Plot drift visualizations of all the features in descending order of the drift scores
        feature_graphs = PlotSortedVisualizations(feature_drift_info)

        html = '<br>' + f'<h2>{mode} Data Drift for {project_name}</h2>'
        
        if cat_drift_info and cat_graph:
            html += '<br>' + '<br>'.join(cat_drift_info) + '<br>' + cat_graph
        if num_drift_info and num_graph:
            html += '<br>' + '<br>'.join(num_drift_info) + '<br>' + num_graph
        if general_drift_info:
            html += '<br>' + '<br>'.join(general_drift_info)
        if feature_graphs:
            html += '<br>' + '<br>'.join(feature_graphs)
        with open(data_drift_view.path, 'w') as f:
            f.write(html)

    else:
        print('No dataset q, possibility of cold start, triggering modelling...')
        drift_status = False

    num_quadrant_dict = json.dumps(num_quadrant_dict)
    cat_quadrant_dict = json.dumps(cat_quadrant_dict)
    feature_drift_dict = json.dumps(feature_drift_dict)

    return (
        dq_status,
        drift_status,
        num_quadrant_dict,
        cat_quadrant_dict,
        feature_drift_dict
    )