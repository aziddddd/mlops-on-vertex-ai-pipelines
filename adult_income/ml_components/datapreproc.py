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

def data_preprocess(
    bucket_name: str,
    category_threshold: int,
    drop_columns: str,
    target_column: str,
    dataset: Input[Dataset],
    processed_dataset: Output[Dataset],
) -> NamedTuple(
    "Outputs",
    [
        ("column_types_str", str),
    ],
):
    from google.cloud import storage
    import pandas as pd
    import numpy as np
    import json
    import os

    df = pd.read_parquet(dataset.path)

    drop_columns = json.loads(drop_columns)
    column_types = {col: 'categorical' if df[col].nunique() <= category_threshold else 'numerical' for col in df.columns}
    column_types = {key: val for (key, val) in column_types.items() if key not in drop_columns}

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name.replace('gs://', ''))

    def preprocess(df):
        # Drop desired column
        df = df.drop(columns=drop_columns)

        #############################################################################################################
        print('Preprocess your data here.')

        # manual encode for target column as a firm step
        df['income_bracket'] = df['income_bracket'].apply(
            lambda row: 1 if row.strip() == '>50K' else 0
        )
        #############################################################################################################

        # automatic categorical encode for xgboost, can comment if dont want to use.
        for col in [col for (col, col_type) in column_types.items() if col_type=='categorical' and col!=target_column]:
            if col in drop_columns:
                continue
            local_dir = f'{col}_cat_mapping.json'
            remote_dir = f'artifacts/{local_dir}'

            blobs = storage_client.list_blobs(
                bucket_name.replace('gs://', ''),
                prefix=remote_dir,
            )
            # check existing cat mapping
            check = False
            for blob in blobs:
                print(blob.name)
                check=True
                break

            print(f'Check for {col} : {check}')
            # use existing cat mapping, update if there's new key
            if check:
                blob = bucket.blob(remote_dir)
                blob.download_to_filename(local_dir)
                with open(local_dir, 'r') as openfile:
                    cat_mapping = json.load(openfile)
                for val in df[col].unique():
                    if str(val) not in cat_mapping:
                        cat_mapping[str(val)] = len(cat_mapping)

            # create new cat mapping
            else:
                cat_mapping = {str(i): idx for (idx, i) in enumerate(df[col].unique())}

            df[col] = df[col].apply(
                lambda row : cat_mapping[str(row)]
            )

            # update cat mapping
            cat_mapping_str = json.dumps(cat_mapping)
            with open(local_dir, 'w') as outfile:
                outfile.write(cat_mapping_str)
            blob = bucket.blob(remote_dir)
            blob.upload_from_filename(local_dir)

        return df

    processed_df = preprocess(df)
    processed_df.to_parquet(processed_dataset.path, index=False)
    column_types_str = json.dumps(column_types)

    return (
        column_types_str,
    )

def data_split(
    auto_balance: str,
    seed: int,
    train_size: float,
    target_column: str,
    column_types_str: str,
    processed_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    x_train_dataset: Output[Dataset],
    x_val_dataset: Output[Dataset],
    y_train_dataset: Output[Dataset],
    y_val_dataset: Output[Dataset],
):
    from sklearn.model_selection import train_test_split
    from datetime import datetime, timedelta
    from collections import Counter
    import pandas as pd
    import json

    processed_df = pd.read_parquet(processed_dataset.path)
    column_types = json.loads(column_types_str)

    x = processed_df[[i for i in processed_df.columns if i != target_column]]
    y = processed_df[target_column]

    categorical_columns = [key for (key, val) in column_types.items() if val=='categorical' and key!=y.name]
    categorical_features_indices = [list(x.columns).index(cat) for cat in categorical_columns]

    if auto_balance:
        sampler_params = {'random_state':seed}
        if auto_balance == 'over_sampling':
            from imblearn.over_sampling import SMOTENC, SMOTE

            if categorical_features_indices:
                sampler_params['categorical_features'] = categorical_features_indices
                sampler = SMOTENC(**sampler_params)
            else:
                sampler = SMOTE(**sampler_params)

        elif auto_balance == 'under_sampling':
            from imblearn.under_sampling import RandomUnderSampler

            sampler = RandomUnderSampler(**sampler_params)

        x, y = sampler.fit_resample(x, y)

    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1-train_size, random_state=seed, stratify=target_column)  ## TypeError: Singleton array array('Class', dtype='<U5') cannot be considered a valid collection.
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1-train_size, random_state=seed)
    
    # Combine back x and y to push to bq
    bq_train = pd.concat(
        [
            x_train.reset_index(drop=True),
            y_train.reset_index(drop=True)
        ],
        axis=1
    )

    print(f'processed_df : {processed_df.shape}')

    print(f'train : {bq_train.shape}')
    print(f'x_train : {x_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'x_val : {x_val.shape}')
    print(f'y_val : {y_val.shape}')
    print(f'x : {x.shape}')
    print(f'y : {y.shape}')
    print(f'Training set distribution : {Counter(y_train)}')

    bq_train.to_parquet(train_dataset.path, index=False)
    x_train.to_parquet(x_train_dataset.path, index=False)
    x_val.to_parquet(x_val_dataset.path, index=False)
    y_train.to_frame().to_parquet(y_train_dataset.path, index=False)
    y_val.to_frame().to_parquet(y_val_dataset.path, index=False)