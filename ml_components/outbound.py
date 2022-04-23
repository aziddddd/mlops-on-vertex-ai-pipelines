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

def generate_bq_table_from_gsc(
    project_id: str,
    dataset_id: str,
    runner: str,
    table_name: str,
    src_run_dt: str,
    dataset_to_save: Input[Dataset],
    dataset_format: str='PARQUET',
    location: str='asia-southeast1',
) -> NamedTuple(
    "Outputs",
    [
        ("bq_path", str),
    ],
):
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(
        project=project_id,
        location=location
    )

    # Configure the external data source
    dataset_ref = bigquery.DatasetReference(
        project_id, 
        dataset_id
    )

    table_id = f'{runner}_{table_name}_{src_run_dt}'
    table = bigquery.Table(dataset_ref.table(table_id))
    external_config = bigquery.ExternalConfig(dataset_format)
    external_config.source_uris = [dataset_to_save.uri]
    external_config.options.skip_leading_rows = 1  # optionally skip header row
    table.external_data_configuration = external_config

    # Create a permanent table linked to the GCS file
    table = client.create_table(
        table=table,
        exists_ok=True,
    )  # API request
    print(f'Table {table_id} has been created')
    print(f'Dataset URI : {dataset_to_save.uri}')
    print(f'Dataset Format : {dataset_format}')
    
    bq_path = f'{dataset_id}.{table_id}'
    return (
        bq_path,
    )

def update_model_league(
    project_id: str,
    job_id: str,
    src_run_dt: str,
    bq_path: str,
    choose_model: str,
    runner: str,
    is_cold_start: bool,
    update_mode: str,
    model_object: Input[Model],
):
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import pandas_gbq
    import json
    import pytz
    import ast

    # Get model league to update
    model_league_path = f'test_tracking.{runner}_model_league'
    all_contestant_query = f"""
    SELECT * FROM `{model_league_path}`
    """
    model_league = pandas_gbq.read_gbq(
        query=all_contestant_query, 
        project_id=project_id,
        use_bqstorage_api=True,
        dtypes={
            'src_run_dt': np.datetime64()
        },
    )
    model_league['pred_bq_paths'] = model_league['pred_bq_paths'].apply(
        lambda row: ast.literal_eval(row)
    )
    model_league['pred_job_ids'] = model_league['pred_job_ids'].apply(
        lambda row: ast.literal_eval(row)
    )

    if update_mode == 'train':
        # if new model wins
        if choose_model == 'challenger':
            model_league['tag'] = 'archived'
            new_model_league = model_league.append(
                {
                    'src_run_dt': pd.Timestamp(
                        datetime.strptime(src_run_dt, "%Y-%m-%dT%H_%M_%S"),
                        tz=pytz.timezone('UTC')
                    ),
                    'tag': 'production',
                    'model_path': model_object.uri,
                    'train_bq_path': bq_path,
                    'train_job_id': job_id,
                    'pred_bq_path': None,
                    'pred_bq_paths': [],
                    'pred_job_id': None,
                    'pred_job_ids': [],
                },
                ignore_index=True
            )
        # if new model loses
        elif choose_model == 'champion':
            if is_cold_start:
                # Update the model league with our new model as champion
                # as there's no champion yet at this point
                new_model_league = model_league.append(
                    {
                        'src_run_dt': pd.Timestamp(
                            datetime.strptime(src_run_dt, "%Y-%m-%dT%H_%M_%S"),
                            tz=pytz.timezone('UTC')
                        ),
                        'tag': 'production',
                        'model_path': model_object.uri,
                        'train_bq_path': bq_path,
                        'train_job_id': job_id,
                        'pred_bq_path': None,
                        'pred_bq_paths': [],
                        'pred_job_id': None,
                        'pred_job_ids': [],
                    },
                    ignore_index=True
                )
            else:
                # Update the model league with our new model as challenger
                new_model_league = model_league.append(
                    {
                        'src_run_dt': pd.Timestamp(
                            datetime.strptime(src_run_dt, "%Y-%m-%dT%H_%M_%S"),
                            tz=pytz.timezone('UTC')
                        ),
                        'tag': 'archived',
                        'model_path': model_object.uri,
                        'train_bq_path': bq_path,
                        'train_job_id': job_id,
                        'pred_bq_path': None,
                        'pred_bq_paths': [],
                        'pred_job_id': None,
                        'pred_job_ids': [],
                    },
                    ignore_index=True
                )

    elif update_mode == 'pred':
        new_model_league = model_league.copy()
        new_model_league.loc[new_model_league['tag']=='production', 'pred_bq_path'] = bq_path
        new_model_league.loc[new_model_league['tag']=='production', 'pred_bq_paths'][0].append(bq_path)
        
        new_model_league.loc[new_model_league['tag']=='production', 'pred_job_id'] = job_id
        new_model_league.loc[new_model_league['tag']=='production', 'pred_job_ids'][0].append(job_id)

    new_model_league['src_run_dt'] = new_model_league['src_run_dt'].dt.tz_localize(None)
    new_model_league['pred_bq_paths'] = new_model_league['pred_bq_paths'].apply(
        lambda row: json.dumps(list(set(row)))
    )
    new_model_league['pred_job_ids'] = new_model_league['pred_job_ids'].apply(
        lambda row: json.dumps(list(set(row)))
    )

    # Push the updated model league
    pandas_gbq.to_gbq(
        new_model_league,
        destination_table=model_league_path,
        project_id=project_id,
        location='us-central1',
        if_exists='replace',
    )

def export_prediction(
    project_id: str,
    runner: str,
    prediction_dataset: Input[Dataset],
):
    import pandas as pd
    import pandas_gbq

    test = pd.read_parquet(prediction_dataset.path)

    print('Perform your export_prediction here.')