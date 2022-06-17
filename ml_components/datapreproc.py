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
    drop_columns: str,
    dataset: Input[Dataset],
    processed_dataset: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    import json

    df = pd.read_parquet(dataset.path)

    drop_columns = json.loads(drop_columns)

    # DERIVE DISTRIBUTION OF EACH FEATURE
    def preprocess(df):
        # Drop time column as it i not used for modelling
        df = df.drop(columns=drop_columns)
        print('Preprocess your data here.')
        return df
    
    processed_df = preprocess(df)

    processed_df.to_parquet(processed_dataset.path, index=False)


def data_split(
    auto_balance: str,
    category_threshold: int,
    seed: int,
    train_size: float,
    target_column: str,
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

    x = processed_df[[i for i in processed_df.columns if i != target_column]]
    y = processed_df[target_column]

    if auto_balance:
        sampler_params = {'random_state':seed}
        if auto_balance == 'over_sampling':
            from imblearn.over_sampling import SMOTENC, SMOTE

            column_types = {col: 'categorical' if processed_df[col].nunique() <= category_threshold else 'numerical' for col in processed_df.columns}
            categorical_columns = [key for (key, val) in column_types.items() if val=='categorical' and key!=y.name]
            categorical_features_indices = [list(x.columns).index(cat) for cat in categorical_columns]

            if categorical_features_indices:
                sampler_params['categorical_features'] = categorical_features_indices
                sampler = SMOTENC(**sampler_params)
            else:
                sampler = SMOTE(**sampler_params)

        elif auto_balance == 'under_sampling':
            from imblearn.under_sampling import RandomUnderSampler

            sampler = RandomUnderSampler(**sampler_params)

        x, y = sampler.fit_resample(x, y)

    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1-train_size, random_state=seed, stratify=target_column)  
    ## TypeError: Singleton array array('Class', dtype='<U5') cannot be considered a valid collection.
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