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
    sku_column: str,
    ts_column: str,
    val_column: str,
    dataset: Input[Dataset],
    processed_dataset: Output[Dataset],
    # processed_dataset_path: OutputPath("Dataset"),
) -> NamedTuple(
    "Outputs",
    [
        ("train_dataset_path", str),
        ("total_tables", int),
    ],
):
    import pandas as pd
    import numpy as np
    import json
    
    print('loading raw dataset...')
    df = pd.read_parquet(dataset.path)

    # Define your own pre-processing
    def preprocess(df):
        df[sku_column] = df[sku_column].astype(str)
        df[val_column] = df[val_column].astype(float)
        df[ts_column]= pd.to_datetime(df[ts_column])

#         def clean_sku_name(row):
#             import re

#             # i.e. haha_20220101
#             regex1 = re.compile(r'^(.*)_[\d]{8}$') # with underscores

#             # i.e. haha20220101
#             regex2 = re.compile(r'^(.*)[\d]{8}$') # without underscores

#             if regex1.match(row):
#                 match_obj = regex1.match(row)
#                 return ''
#                 # return match_obj.group(1)

#             elif regex2.match(row):
#                 match_obj = regex2.match(row)
#                 return ''
#                 # return match_obj.group(1)

#             else:
#                 return row
#         df[sku_column] = df[sku_column].apply(
#             lambda row: clean_sku_name(row)
#         )
        df = df[df[sku_column]!='']
        return df

    print('preprocess raw dataset...')
    processed_df = preprocess(df)

    print('retrieve tables metadata...')
    train_dataset_path = json.dumps(
        [
            {'sku_name': sku_name}  for sku_name in processed_df[sku_column].unique()
        ]
    )
    total_tables = len(train_dataset_path)

    processed_df.to_parquet(processed_dataset.path, index=False)

    return (
        train_dataset_path,
        total_tables,
    )

def data_interpolate(
    sku_name: str,
    train_start_date: str,
    train_end_date: str,
    freq: str,
    sku_column: str,
    ts_column: str,
    val_column: str,
    processed_dataset: Input[Dataset],
    interpolated_dataset: Output[Dataset],
):
    from darts.utils.missing_values import fill_missing_values
    from darts import TimeSeries
    import pandas as pd
    import numpy as np

    processed_df = pd.read_parquet(processed_dataset.path)
    
    sku_df = processed_df[processed_df[sku_column]==sku_name]

    sku_df = sku_df.sort_values(by=ts_column)

    sku_df = sku_df.drop_duplicates(subset=ts_column, keep='last')

    if sku_df[ts_column].tolist()[0].strftime('%Y-%m-%d') == train_start_date:
        start_date = train_start_date
    else:
        start_date = sku_df[ts_column].tolist()[0].strftime('%Y-%m-%d')

    date_reindex = pd.date_range(
        start=start_date,
        end=train_end_date,
        freq=freq
    )
    sku_df = sku_df.set_index(ts_column)
    sku_df = sku_df.reindex(date_reindex)

    sku_df[sku_column] =  sku_name
    sku_df = sku_df.reset_index().rename(columns={'index': ts_column})

    # Fill mising TS data with pandas df interpolation with method 'time' for ts.
    sku_df[val_column] = fill_missing_values(
        TimeSeries.from_dataframe(
            sku_df,
            ts_column, 
            val_column
        ),
        method='time'
    ).pd_dataframe().reset_index()[val_column]

    sku_df.to_parquet(interpolated_dataset.path, index=False)