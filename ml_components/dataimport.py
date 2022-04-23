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

def get_rundates(
    run_date: str,
) -> NamedTuple(
    "Outputs",
    [
        ("usage_run_date", str),
        ("src_run_dt", str),
    ],
):
    from datetime import datetime, timedelta
    import calendar

    src_run_dt = datetime.strftime(
        datetime.now(), # + timedelta(hours=8),
        "%Y-%m-%dT%H_%M_%S"
    )

    if run_date:
        print('The rundates are provided, using them now...')

        usage_run_date = run_date
    else:
        print('The rundates are not fully provided, self-inferring them now...')

        now = datetime.now()
        run_date_obj = now.replace(day=1)
        usage_run_date = run_date_obj.strftime('%Y-%m-%d')
    
    return (
        usage_run_date,
        src_run_dt,
    )

def get_period(
    pred_month_start: str
) -> NamedTuple(
    "Outputs",
    [
        ("pred_month", str),
        ("train_months", str),
        ("prev_pred_month", str),
        ("prev_train_months", str),
    ],
):
    from datetime import datetime
    import pandas as pd
    import json

    pred_month_start_obj = datetime.strptime(pred_month_start, '%Y-%m-%d')
    
    prev_train_mth1 = datetime.strftime(pred_month_start_obj - pd.DateOffset(months=5), '%Y-%m-%d')
    prev_train_mth2 = datetime.strftime(pred_month_start_obj - pd.DateOffset(months=4), '%Y-%m-%d')
    prev_train_mth3 = datetime.strftime(pred_month_start_obj - pd.DateOffset(months=3), '%Y-%m-%d')
    train_mth1 = datetime.strftime(pred_month_start_obj - pd.DateOffset(months=4), '%Y-%m-%d')
    train_mth2 = datetime.strftime(pred_month_start_obj - pd.DateOffset(months=3), '%Y-%m-%d')
    train_mth3 = datetime.strftime(pred_month_start_obj - pd.DateOffset(months=2), '%Y-%m-%d')

    prev_train_mths = json.dumps(
        [
            prev_train_mth1,
            prev_train_mth2,
            prev_train_mth3,
        ]
    )
    train_mths = json.dumps(
        [
            train_mth1,
            train_mth2,
            train_mth3,
        ]
    )

    
    prev_pred_mth = datetime.strftime(pred_month_start_obj - pd.DateOffset(months=1), '%Y-%m-%d')
    pred_mth = pred_month_start

    print("Train on: ", train_mths)
    print("Predict on: ", pred_mth)
    print("Previous Train on: ", prev_train_mths)
    print("Previous Predict on: ", prev_pred_mth)

    return (
        pred_mth,
        train_mths,
        prev_pred_mth,
        prev_train_mths,
    )

def get_import_query(
    months: str,
) -> NamedTuple(
    "Outputs",
    [
        ("query", str),
    ],
):
    import json

    try:
        months_list = json.loads(months)
    except json.decoder.JSONDecodeError as err:
        months_list = [months]
    months_list_str = ", ".join([f'"{mon}"' for mon in months_list])

    query = f"""
    (
        SELECT * FROM `tfx-oss-public.palmer_penguins.palmer_penguins`
        WHERE species = 0
        LIMIT 5
    )
    union all 
    (
        SELECT * FROM `tfx-oss-public.palmer_penguins.palmer_penguins`
        WHERE species = 1
        LIMIT 5
    )
    """

    return (
        query,
    )

def bq_query_no_return(
    project_id:str,
    query:str,
):
    import google.cloud.bigquery as bq
    bq_client = bq.Client(project=project_id)
    bq_job = bq_client.query(query)
    bq_job.result() # Wait for BQ to finish execute

def bq_query_to_dataframe(
    project_id: str,
    query: str,
    dataset: Output[Dataset],
):
    import google.cloud.bigquery as bq
    import pandas as pd
    
    bq_client = bq.Client(project=project_id)
    bq_job = bq_client.query(query)
    bq_job.result() # Wait for BQ to finish execute
    query_result = bq_job.to_dataframe()
    
    dataset.metadata["query"] = query
    query_result.to_parquet(dataset.path, index=False)

def get_previous_dataset_from_model_league(
    project_id: str,
    dataset_type: str, # train/pred
    runner: str,
    previous_dataset: Output[Dataset],
):
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import pandas_gbq

    # Get model league
    model_league_path = f'test_tracking.{runner}_model_league'
    production_query = f"""
    SELECT * FROM `{model_league_path}`
    """
    model_league = pandas_gbq.read_gbq(
        query=production_query, 
        project_id=project_id,
        use_bqstorage_api=True,
        dtypes={
            'src_run_dt': np.datetime64()
        },
    )

    if len(model_league) > 0:
        model_league = model_league[model_league['tag']=='production']
        dataset_path = model_league[f'{dataset_type}_bq_path'].tolist()[0]

        if not dataset_path:
            previous_df = pd.DataFrame()
            previous_df.to_parquet(previous_dataset.path, index=False)
        else:
            # Get previous dataset
            previous_dataset_query = f"""
            SELECT * FROM `{dataset_path}`
            """

            previous_df = pandas_gbq.read_gbq(
                query=previous_dataset_query, 
                project_id=project_id,
                use_bqstorage_api=True,
            )
            previous_df.to_parquet(previous_dataset.path, index=False)

    # cold start
    else:
        previous_df = pd.DataFrame()
        previous_df.to_parquet(previous_dataset.path, index=False)
    
    print(f'previous_df : {previous_df.shape}')