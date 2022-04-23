# from kfp_components import datapreproc
# from kfp.v2.dsl import (
#     Dataset,
# )
# from unittest.mock import patch
# from unittest import mock
# import pytest

# def test_data_preprocess(mocker):
#     import pandas as pd

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as dataset_obj:
#         dataset = dataset_obj
#         dataset.path = 'tests/unit/resources/datapreproc/data_preprocess_input.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as processed_dataset_obj:
#         processed_dataset = processed_dataset_obj
#         processed_dataset.path = 'tests/unit/resources/datapreproc/data_preprocess_output.parquet'
#         processed_dataset.metadata = {}

#     # this will fail if preprocessing is different.
#     datapreproc.data_preprocess(
#         dataset=dataset,
#         processed_dataset=processed_dataset,
#     )

# def test_data_split(mocker):
#     project_id = 'test-project'
#     train_months = '["2021-05-01", "2021-06-01", "2021-07-01"]'
#     runner = 'dev'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as processed_dataset_obj:
#         processed_dataset = processed_dataset_obj
#         processed_dataset.path = 'tests/unit/resources/datapreproc/data_split_processed_dataset_input.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as train_dataset_obj:
#         train_dataset = train_dataset_obj
#         train_dataset.path = 'tests/unit/resources/datapreproc/data_split_train_dataset_output.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as x_train_dataset_obj:
#         x_train_dataset = x_train_dataset_obj
#         x_train_dataset.path = 'tests/unit/resources/datapreproc/data_split_x_train_dataset_output.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as x_val_dataset_obj:
#         x_val_dataset = x_val_dataset_obj
#         x_val_dataset.path = 'tests/unit/resources/datapreproc/data_split_x_val_dataset_output.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as y_train_datasetobj:
#         y_train_dataset = y_train_datasetobj
#         y_train_dataset.path = 'tests/unit/resources/datapreproc/data_split_y_train_dataset_output.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as y_val_dataset_obj:
#         y_val_dataset = y_val_dataset_obj
#         y_val_dataset.path = 'tests/unit/resources/datapreproc/data_split_y_val_dataset_output.parquet'

#     # this will fail if preprocessing is different.
#     datapreproc.data_split(
#         train_months=train_months,
#         processed_dataset=processed_dataset,
#         train_dataset=train_dataset,
#         x_train_dataset=x_train_dataset,
#         x_val_dataset=x_val_dataset,
#         y_train_dataset=y_train_dataset,
#         y_val_dataset=y_val_dataset,
#     )