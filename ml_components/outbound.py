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

def data_collection(
    bucket_name: str,
    src_run_dt: str,
    mode: str,
    grand_dataset: Output[Dataset],
):

    # Download from GCS to local(container) and then concat.
    import subprocess

    subprocess.check_output(
        [
            "gsutil",
            "cp",
            "-r",
            f"{bucket_name}/artifacts/{mode}/{src_run_dt}/",
            "."
        ]
    )

    from pathlib import Path
    import pandas as pd

    data_dir = Path(f'{src_run_dt}')
    output_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in data_dir.glob('*.parquet')
    )
    output_df.to_parquet(grand_dataset.path, index=False)


def html_collection(
    bucket_name: str,
    src_run_dt: str,
    datadrift_view: Output[HTML],
    modelling_view: Output[HTML],
    prediction_view: Output[HTML],
    
):

    # Download from GCS to local(container) and then concat.
    from google.cloud import storage
    import subprocess
    import os

    storage_client = storage.Client()

    modes = {
        'datadrift': datadrift_view.path,
        'modelling': modelling_view.path,
        'prediction': prediction_view.path,
    }
    for (mode, html_path) in modes.items():
        blobs = storage_client.list_blobs(
            bucket_name.replace('gs://', ''),
            prefix=f'artifacts/html/{mode}/{src_run_dt}',
        )
        check = False
        for blob in blobs:
            # check if any html exists in folder
            print(blob.name)
            check=True
            break

        print(f'Check for {mode} : {check}')
        if check:
            if not os.path.exists(f'{mode}/'):
                os.makedirs(f'{mode}/')

            subprocess.check_output(
                [
                    "gsutil",
                    "cp",
                    "-r",
                    f"{bucket_name}/artifacts/html/{mode}/{src_run_dt}/",
                    f'{mode}/'
                ]
            )

            os.system(f"pandoc -s {mode}/{src_run_dt}/*.html -o output.html")

            subprocess.check_output(
                [
                    "cp",
                    "output.html",
                    html_path,
                ]
            )


def generate_bq_table_from_gsc(
    project_id: str,
    project_name: str,
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

    if src_run_dt:
        table_id = f'{runner}_{table_name}_{project_name}_{src_run_dt}'
    else:
        table_id = f'{runner}_{table_name}_{project_name}'

    table = bigquery.Table(dataset_ref.table(table_id))

    existing_tables = [i.table_id for i in client.list_tables(dataset_id)]
    if table_id in existing_tables:
        # Delete old table if exist
        client.delete_table(
            table=table,
        )

    external_config = bigquery.ExternalConfig(dataset_format)
    external_config.source_uris = [dataset_to_save.uri]
    external_config.options.skip_leading_rows = 1  # optionally skip header row
    table.external_data_configuration = external_config

    # Create a new table linked to the GCS file
    table = client.create_table(
        table=table,
        exists_ok=True,
        retry=bigquery.DEFAULT_RETRY.with_deadline(30)
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
    project_name: str,
    location: str,
    job_id: str,
    sku_column: str,
    metrics_name: str,
    src_run_dt: str,
    train_bq_path: str,
    val_bq_path: str,
    prediction_bq_path: str,
    runner: str,
    tracking_cutoff: int,
    accuracy_board_dataset: Input[Dataset],
):
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import pandas_gbq
    import json
    import pytz
    import ast


    board = pd.read_parquet(accuracy_board_dataset.path)

    # Get the best model for each sku
    board_agg = board.groupby(
        [
            sku_column,
        ]
    ).agg(
        {
            metrics_name: [
                'min',
            ],
        },
    )#.reset_index()

    board_agg.columns = [col[0] for col in board_agg.columns.values]

    board_winner = pd.merge(
        board_agg,
        board,
        how='left',
        on=[
            sku_column,
            metrics_name
        ]
    ).drop_duplicates(
        subset=sku_column,
    )
    board_winner = board_winner[[sku_column, 'model'] + sorted([i for i in board_winner.columns if i not in [sku_column, 'model']])]

    board_winner['src_run_dt'] = pd.Timestamp(
        datetime.strptime(src_run_dt, "%Y-%m-%dT%H_%M_%S"),
        tz=pytz.timezone('UTC')
    )
    board_winner['tag'] = 'production'
    board_winner['train_bq_path'] = train_bq_path
    board_winner['val_bq_path'] = val_bq_path
    board_winner['prediction_bq_path'] = prediction_bq_path
    board_winner['job_id'] = job_id
    board_winner = board_winner.rename(columns={sku_column: 'sku'})

    # Get model league to update
    model_league_path = f'MLOPS_TRACKING.{runner}_model_league_{project_name}'
    all_contestant_query = f"""
    SELECT * FROM `{model_league_path}`
    """
    model_league = pandas_gbq.read_gbq(
        query=all_contestant_query, 
        project_id=project_id,
        use_bqstorage_api=True,
        location=location,
        dtypes={
            'src_run_dt': np.datetime64()
        },
    )

    # send old production to archived if have new champion
    model_league.loc[model_league['sku'].isin(board_winner['sku'].tolist()), 'tag'] = 'archived'

    new_model_league = pd.concat(
        [
            board_winner,
            model_league
        ],
        ignore_index=True
    )
    new_model_league['src_run_dt'] = new_model_league['src_run_dt'].dt.tz_localize(None)

    # keep only N recent pipeline runs per SKU to reduce model league grow oversized.
    new_model_league = new_model_league.groupby(
        [
            'sku',
        ]
    ).head(tracking_cutoff).sort_values(by=['sku','src_run_dt'], ascending=[True, False]).reset_index(drop=True)

    # Push the updated model league
    pandas_gbq.to_gbq(
        new_model_league,
        destination_table=model_league_path,
        project_id=project_id,
        location=location,
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