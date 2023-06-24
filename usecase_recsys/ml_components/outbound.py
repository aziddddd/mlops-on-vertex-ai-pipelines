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
    src_run_dt: str,
    user_bq_path: str,
    catalog_bq_path: str,
    train_bq_path: str,
    chosen_embedding_dimension: int,
    chosen_model: str,
    runner: str,
    commit_short_sha: str,
    is_cold_start: bool,
    is_endpoint: bool,
    mlops_topic: str,
    chosen_unique_catalog_ids_dataset: Input[Dataset],
    chosen_unique_user_ids_dataset: Input[Dataset],
    cg_model_object: Input[Model],
    cg_index_object: Input[Artifact],
    r_model_object: Input[Model],
):
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import pandas_gbq
    import json
    import pytz
    import ast

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

    with open(chosen_unique_catalog_ids_dataset.path, 'r') as openfile: chosen_unique_catalog_ids = json.load(openfile);
    with open(chosen_unique_user_ids_dataset.path, 'r') as openfile: chosen_unique_user_ids = json.load(openfile);

    entry = {
        'src_run_dt': pd.Timestamp(
            datetime.strptime(src_run_dt, "%Y-%m-%dT%H_%M_%S"),
            tz=pytz.timezone('UTC')
        ),
        'tag': 'production',
        'commit_short_sha': commit_short_sha,
        'user_bq_path': user_bq_path,
        'catalog_bq_path': catalog_bq_path,
        'train_bq_path': train_bq_path,
        'cg_model_path': cg_model_object.uri,
        'cg_index_path': cg_index_object.uri,
        'r_model_path': r_model_object.uri,
        'job_id': job_id,
        'unique_catalog_ids': chosen_unique_catalog_ids,
        'unique_user_ids': chosen_unique_user_ids,
        'embedding_dimension': chosen_embedding_dimension,
    }
    # if new model wins
    if chosen_model == 'challenger':
        model_league['tag'] = 'archived'
        new_model_league = model_league.append(entry, ignore_index=True)

        if is_endpoint:
            # Triggering model refresh on endpoint
            from concurrent import futures
            from typing import Callable
            from google.cloud import pubsub_v1

            trigger_info = json.dumps(
                {
                    'mode': 'recsys',
                    'usecase_id': project_name,
                    'cg_model_path': cg_model_object.uri,
                    'cg_index_path': cg_index_object.uri,
                    'r_model_path': r_model_object.uri,
                }
            )

            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path('your-project-id', mlops_topic.split('/')[-1])
            publish_futures = []

            def get_callback(
                publish_future: pubsub_v1.publisher.futures.Future, 
                data: str,
            ) -> Callable[[pubsub_v1.publisher.futures.Future], None]:
                def callback(publish_future: pubsub_v1.publisher.futures.Future) -> None:
                    try:
                        # Wait 60 seconds for the publish call to succeed.
                        print(publish_future.result(timeout=60))
                    except futures.TimeoutError:
                        print(f"Publishing {data} timed out.")
                return callback

            # When you publish a message, the client returns a future.
            publish_future = publisher.publish(topic_path, trigger_info.encode("utf-8"))
            # Non-blocking. Publish failures are handled in the callback function.
            publish_future.add_done_callback(get_callback(publish_future, trigger_info))
            publish_futures.append(publish_future)

            # Wait for all the publish futures to resolve before exiting.
            futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)
            print(f"Published messages with error handler to {topic_path}.")

    # if new model loses
    elif chosen_model == 'champion':
        if is_cold_start:
            # Update the model league with our new model as champion
            # as there's no champion yet at this point
            new_model_league = model_league.append(entry, ignore_index=True)
        else:
            entry['tag'] = 'archived'
            # Update the model league with our new model as challenger
            new_model_league = model_league.append(entry, ignore_index=True)

    new_model_league['src_run_dt'] = new_model_league['src_run_dt'].dt.tz_localize(None)

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