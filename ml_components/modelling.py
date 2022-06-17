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

def model_trainer(
    seed: int,
    model_params: str,
    x_train_dataset: Input[Dataset],
    x_val_dataset: Input[Dataset],
    y_train_dataset: Input[Dataset],
    y_val_dataset: Input[Dataset],
    model_object: Output[Model],
):

    import xgboost as xgb
    import pandas as pd
    import json

    x_train = pd.read_parquet(x_train_dataset.path)
    y_train = pd.read_parquet(y_train_dataset.path)
    x_val = pd.read_parquet(x_val_dataset.path)
    y_val = pd.read_parquet(y_val_dataset.path)
    model_params_dict = json.loads(model_params)
    model_params_dict['random_state'] = seed

    # Do your model training here.
    model = xgb.XGBClassifier(**model_params_dict)
    model.fit(
        x_train,
        y_train,
        eval_set=[
            (
                x_val,
                y_val
            )
        ],
        eval_metric='auc',
        verbose=False,
        early_stopping_rounds=20
    )

    model_results = model.evals_result()
    model.save_model(model_object.path)

def get_champion_model(
    project_id: str,
    project_name: str,
    runner:str,
    location: str,
    model_champion_object: Output[Model],
) -> NamedTuple(
    "Outputs",
    [
        ("is_cold_start", bool),
    ],
):
    from google.cloud import storage
    import xgboost as xgb
    import pandas_gbq

    model_league_path = f'MLOPS_TRACKING.{runner}_model_league_{project_name}'
    production_query = f"""
    SELECT * FROM `{model_league_path}`
    """
    model_league = pandas_gbq.read_gbq(
        query=production_query, 
        project_id=project_id,
        use_bqstorage_api=True,
        location=location,
    )
    
    if len(model_league) > 0:
        model_league = model_league[model_league['tag']=='production'].reset_index()
        storage_client = storage.Client()
        model_path = model_league['model_path'][0].replace('gs://', '')
        bucket = storage_client.get_bucket(model_path.split('/')[0])
        blob = bucket.blob('/'.join(model_path.split('/')[1:]))
        blob.download_to_filename(model_path.split('/')[-1])

        champion_model = xgb.XGBClassifier()
        champion_model.load_model(model_path.split('/')[-1])
        champion_model.save_model(model_champion_object.path)
        
        is_cold_start = False
    else:
        is_cold_start = True
    
    return (
        is_cold_start,
    )

def model_evaluator(
    is_cold_start: bool,
    model_object: Input[Model],
    model_champion_object: Input[Model],
    x_train_dataset: Input[Dataset],
    x_val_dataset: Input[Dataset],
    y_val_dataset: Input[Dataset],
    feature_importance_view: Output[HTML],
    metrics_object: Output[Metrics],
    model_eval_score_object: Output[Dataset],
    model_champion_eval_score_object: Output[Dataset],
) -> NamedTuple(
    "Outputs",
    [
        ("best_model", str),
    ],
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import xgboost as xgb
    import pandas as pd
    import numpy as np
    import sklearn
    import base64
    import shap
    import io

    # needed for feature importance html view
    x_train = pd.read_parquet(x_train_dataset.path)

    x_val = pd.read_parquet(x_val_dataset.path)
    y_val = pd.read_parquet(y_val_dataset.path)

    # Declare xgb model objects
    model, model_champion = xgb.XGBClassifier(), xgb.XGBClassifier()

    # Load models
    model.load_model(model_object.path)
    if not is_cold_start:
        model_champion.load_model(model_champion_object.path)

    # Start capturing graphs using io
    def plot_to_html(tmpfile):
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        plot_as_html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        return plot_as_html

    try:
        feature_importance_array = model.feature_importances_
    except TypeError as err:
        feature_importance_array = np.zeros(len(x_train.columns))

    feature_imp = pd.DataFrame(
        sorted(
            zip(
                feature_importance_array,
                x_train.columns
            )
        ),
        columns=[
            'Value',
            'Feature'
        ]
    )
    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="Value", 
        y="Feature", 
        data=feature_imp.sort_values(
            by="Value",
            ascending=False
        )
    )
    plt.title('XGBoost Features (avg over folds)')
    plt.tight_layout()
    graph_1 = plot_to_html(io.BytesIO())

    plt.figure(figsize=(20, 10))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_val)
    shap.summary_plot(shap_values, x_val, plot_type='dot')
    graph_2 = plot_to_html(io.BytesIO())

    html = graph_1 + '<br>' + graph_2
    with open(feature_importance_view.path, 'w') as f:
        f.write(html)

    # Function to get metrics for the models
    def get_metrics(
        model, 
        X, y, 
        mode='train',
        is_cold_start=False,
    ):
        if is_cold_start:
            metrics  = {
                f'{mode}_Precision': 0.0,
                f'{mode}_Recall': 0.0,
                f'{mode}_AUC': 0.0,
                f'{mode}_GINI': 0.0,
            }

            score_df = pd.DataFrame()
        else:
            predicted_y_score_raw = model.predict_proba(X)
            predicted_y_score = list(map(lambda x: x[1], predicted_y_score_raw))
            predicted_y_churn = list(map(lambda x: 1 if x >= 0.9 else 0, predicted_y_score))
            precision = sklearn.metrics.precision_score(y, predicted_y_churn)
            recall = sklearn.metrics.recall_score(y, predicted_y_churn)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, predicted_y_score)
            auc = sklearn.metrics.auc(fpr, tpr)
            gini = auc * 2 - 1
            metrics  = {
                f'{mode}_Precision': precision,
                f'{mode}_Recall': recall,
                f'{mode}_AUC': auc,
                f'{mode}_GINI': gini,
            }

            score_df = pd.DataFrame(
                {
                    'predicted_y_score': predicted_y_score,
                }
            )
        return score_df, metrics

    model_eval_score_df, model_eval_metrics = get_metrics(model, x_val, y_val, mode='eval')
    model_champion_eval_score_df, model_champion_eval_metrics = get_metrics(model_champion, x_val, y_val, mode='eval', is_cold_start=is_cold_start)

    # Tag metrics to models' resources
    for metric, score in model_eval_metrics.items(): metrics_object.log_metric(f'challenger_{metric}', score);
    for metric, score in model_champion_eval_metrics.items(): metrics_object.log_metric(f'champion_{metric}', score);

    # Save models eval score for model output drift check
    model_eval_score_df.to_parquet(model_eval_score_object.path, index=False)
    model_champion_eval_score_df.to_parquet(model_champion_eval_score_object.path, index=False)

    # Compare and select best model
    eval_comparison = []
    for metric, score in model_eval_metrics.items():
        if score > model_champion_eval_metrics[metric]:
            is_new_model_win = True
        else:
            is_new_model_win = False
        eval_comparison.append(is_new_model_win)

    # if new model wins
    if all(eval_comparison):
        best_model = 'challenger'

    # if new model loses
    else:
        best_model = 'champion'

    return (
        best_model,
    )
