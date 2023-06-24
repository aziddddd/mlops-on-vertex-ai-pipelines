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
        usage_run_date = now.strftime('%Y-%m-%d')
    
    return (
        usage_run_date,
        src_run_dt,
    )

def get_period(
    run_date: str,
    training_months: int,
) -> NamedTuple(
    "Outputs",
    [
        ("train_start_date", str),
        ("train_end_date", str),
        ("pred_start_date", str),
        ("pred_end_date", str),
    ],
):
    from datetime import datetime
    import pandas as pd

    run_date_obj = datetime.strptime(run_date, '%Y-%m-%d')

    train_start_date = datetime.strftime(datetime.strptime(run_date, '%Y-%m-%d') - pd.DateOffset(months=training_months), '%Y-%m-%d')
    train_end_date = run_date

    pred_start_date = datetime.strftime(datetime.strptime(run_date, '%Y-%m-%d') + pd.DateOffset(days=1), '%Y-%m-%d')
    pred_end_date = datetime.strftime(datetime.strptime(pred_start_date, '%Y-%m-%d') + pd.DateOffset(months=3), '%Y-%m-%d')

    return (
        train_start_date,
        train_end_date,
        pred_start_date,
        pred_end_date,
    )

def get_import_query(
    start_date: str,
    end_date: str,
) -> NamedTuple(
    "Outputs",
    [
        ("query", str),
    ],
):
    query = f"""
    SELECT
        snapshot_date,
        CONCAT(
            project_id, '.', 
            dataset_id, '.',
            table_id
        ) AS table_name,
        row_count,
    FROM `your-project-id.TEMP.timeseries_table`
    WHERE snapshot_date BETWEEN '{start_date}' AND '{end_date}'
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
    runner: str,
    sku_column: str,
    ts_column: str,
    val_column: str,
    sku_name: str,
    location: str,
    previous_train_dataset: Output[Dataset],
    previous_val_dataset: Output[Dataset],
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
        train_dataset_path = model_league['train_bq_path'].tolist()[0]
        val_dataset_path = model_league['val_bq_path'].tolist()[0]

        previous_train_query = f"""
        SELECT {sku_column}, {ts_column}, {val_column} FROM `{train_dataset_path}`
        WHERE {sku_column} = '{sku_name}'
        """
        
        previous_train_df = pandas_gbq.read_gbq(
            query=previous_train_query, 
            project_id=project_id,
            use_bqstorage_api=True,
            location=location,
        )

        previous_val_query = f"""
        SELECT {sku_column}, {ts_column}, {val_column} FROM `{val_dataset_path}`
        WHERE {sku_column} = '{sku_name}'
        """

        previous_val_df = pandas_gbq.read_gbq(
            query=previous_val_query, 
            project_id=project_id,
            use_bqstorage_api=True,
            location=location,
        )

        previous_df = pd.concat(
            [
                previous_train_df,
                previous_val_df
            ],
            ignore_index=True
        )

        previous_train_dataset.metadata["query"] = previous_train_query
        previous_val_dataset.metadata["query"] = previous_val_query

        previous_train_df.to_parquet(previous_train_dataset.path, index=False)
        previous_val_df.to_parquet(previous_val_dataset.path, index=False)
        previous_df.to_parquet(previous_dataset.path, index=False)

    # cold start
    else:
        previous_train_df, previous_val_df, previous_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        previous_train_df.to_parquet(previous_train_dataset.path, index=False)
        previous_val_df.to_parquet(previous_val_dataset.path, index=False)
        previous_df.to_parquet(previous_dataset.path, index=False)

    print(f'previous_train_df : {previous_train_df.shape}')
    print(f'previous_val_df : {previous_val_df.shape}')
    print(f'previous_df : {previous_df.shape}')