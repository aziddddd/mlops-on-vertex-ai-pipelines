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
        run_date_obj = now
        usage_run_date = run_date_obj.strftime('%Y-%m-%d')
    
    return (
        usage_run_date,
        src_run_dt,
    )

def get_period(
    run_date: str
) -> NamedTuple(
    "Outputs",
    [
        ("pred_dt", str),
        ("train_dt", str),
    ],
):
    from datetime import datetime, timedelta

    run_date_obj = datetime.strptime(run_date, '%Y-%m-%d')

    # user to define their train and pred months
    train_dt = datetime.strftime(run_date_obj - timedelta(days=1), '%Y-%m-%d')
    pred_dt = run_date

    print("Train on: ", train_dt)
    print("Predict on: ", pred_dt)

    return (
        pred_dt,
        train_dt
    )

def get_import_query(
    datestr: str,
) -> NamedTuple(
    "Outputs",
    [
        ("query", str),
    ],
):
    query = f"""
    select * except(functional_weight, capital_gain, capital_loss, education) from `your-project-id.TEMP.mock__audit_income` a
    where dt = '{datestr}'
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
    project_name: str,
    dataset_type: str, # train/pred
    runner: str,
    location: str,
    previous_dataset: Output[Dataset],
):
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import pandas_gbq

    # Get model league
    model_league_path = f'MLOPS_TRACKING.{runner}_model_league_{project_name}'
    production_query = f"""
    SELECT * FROM `{model_league_path}`
    """
    model_league = pandas_gbq.read_gbq(
        query=production_query, 
        project_id=project_id,
        use_bqstorage_api=True,
        location=location,
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