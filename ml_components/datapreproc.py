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
    dataset: Input[Dataset],
    processed_dataset: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    df = pd.read_parquet(dataset.path)

    # DERIVE DISTRIBUTION OF EACH FEATURE
    def preprocess(df):
        print('Preprocess your data here.')
        return df
    
    processed_df = preprocess(df)
    
    # L2 QC
    l2_qc = processed_df.isnull().sum()
    print("To find null in dataset after pre-processing:\n", l2_qc)
    
    for feature, null_count in l2_qc.iteritems():
        processed_dataset.metadata[feature] = null_count

    processed_df.to_parquet(processed_dataset.path, index=False)
    

def data_split(
    train_months: str,
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
    
    seed = 0

    processed_df = pd.read_parquet(processed_dataset.path)
    train_months_list = json.loads(train_months)

    x = processed_df[[i for i in processed_df.columns if i != 'species']]
    y = processed_df['species']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    
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