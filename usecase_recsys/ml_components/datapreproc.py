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
    numerical_columns: str,
    dataset: Input[Dataset],
    processed_dataset: Output[Dataset],
):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np
    import json

    df = pd.read_parquet(dataset.path)

    drop_columns = json.loads(drop_columns)
    numerical_columns = json.loads(numerical_columns)

    # DERIVE DISTRIBUTION OF EACH FEATURE
    def preprocess(df):
        # Drop time column as it i not used for modelling
        df = df.drop(columns=drop_columns)
        print('Preprocess your data here.')

        for col in numerical_columns:
            scl = StandardScaler()
            df[f'scaled_{col}'] = scl.fit_transform(df[col].values.reshape(-1,1))

        df = df.drop(columns=numerical_columns)
        df = df.rename(columns={f'scaled_{col}': col for col in numerical_columns})

        return df
    
    processed_df = preprocess(df)

    processed_df.to_parquet(processed_dataset.path, index=False)


def data_split(
    seed: int,
    train_size: float,
    processed_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    from sklearn.model_selection import train_test_split
    from datetime import datetime, timedelta
    from collections import Counter
    import pandas as pd
    import json

    processed_df = pd.read_parquet(processed_dataset.path)

    train, test = train_test_split(
        processed_df, 
        test_size=1-train_size, 
        random_state=seed
    )
    print(f'processed_df : {processed_df.shape}')
    print(f'train : {train.shape}')
    print(f'test : {test.shape}')

    train.to_parquet(train_dataset.path, index=False)
    test.to_parquet(test_dataset.path, index=False)