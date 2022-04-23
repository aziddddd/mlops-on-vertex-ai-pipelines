from kfp_components import dataimport
from kfp.v2.dsl import (
    Dataset,
)
from unittest.mock import patch
from unittest import mock
import pytest
import numpy

def test_get_rundates():
    from datetime import datetime

    # Case 1 : rundates provided
    bcdate = '2021-01-01'
    
    (usage_bcdate, src_run_dt) = dataimport.get_rundates(
        bcdate=bcdate,
    )
    assert(usage_bcdate == bcdate), f'usage_bcdate [{usage_bcdate}] should be same as bcdate [{bcdate}] when bcdate is provided.'

    # Case 2 : rundates provided
    from datetime import datetime, timedelta
    import calendar

    now = datetime.now()
    bcdate = now.replace(day=1)
    
    expected_bcdate = bcdate.strftime('%Y-%m-%d')

    (usage_bcdate, src_run_dt) = dataimport.get_rundates(
        bcdate='',
    )
    assert(usage_bcdate == expected_bcdate), f'usage_bcdate [{usage_bcdate}] should be same as expected_bcdate [{expected_bcdate}] when bcdate is not provided.'


    date_format  = "%Y-%m-%dT%H_%M_%S"
    assert(datetime.strftime(datetime.strptime(src_run_dt, date_format), date_format) == src_run_dt), "Incorrect data format, should be %Y-%m-%dT%H_%M_%S"

def test_get_period():
    pred_month_start = '2021-09-01'

    expected_prev_train_months = '["2021-04-01", "2021-05-01", "2021-06-01"]'
    expected_train_months = '["2021-05-01", "2021-06-01", "2021-07-01"]'
    expected_prev_pred_month = "2021-08-01"
    expected_pred_month = "2021-09-01"

    (pred_month, train_months, prev_pred_month, prev_train_months) = dataimport.get_period(
        pred_month_start=pred_month_start,
    )

    assert(prev_train_months == expected_prev_train_months), 'the retrieved prev_train_months is incorrect.'
    assert(train_months == expected_train_months), 'the retrieved train_months is incorrect.'
    assert(prev_pred_month == expected_prev_pred_month), 'prev_pred_month should be one month behind the provided pred_month_start.'
    assert(pred_month == expected_pred_month), 'pred_month should be same as the provided pred_month_start.'

def test_get_import_query():
    import json

    # Case 1 : json dumped list as input
    months = '["2021-05-01", "2021-06-01", "2021-07-01"]'

    expected_query = f"""
    SELECT * FROM `tfx-oss-public.palmer_penguins.palmer_penguins` LIMIT 1000
    """
    (query,) = dataimport.get_import_query(
        months=months,
    )
    assert(query == expected_query), 'the retrieved query is incorrect.'

    # Case 2 : str as input
    months = "2021-05-01"
    expected_query = f"""
    SELECT * FROM `tfx-oss-public.palmer_penguins.palmer_penguins` LIMIT 1000
    """
    (query,) = dataimport.get_import_query(
        months=months,
    )
    assert(query == expected_query), 'the retrieved query is incorrect.'

def test_bq_query_no_return(mocker):
    import google.cloud.bigquery as bq
    mock_table = mocker.patch('google.cloud.bigquery.Table', autospec=True)
    mock_client = mocker.patch('google.cloud.bigquery.Client', autospec=True)

    client = bq.Client()
    project_id = 'project'
    dataset_id = 'dataset'
    table_id = 'table'
    table_path = f'{project_id}.{dataset_id}.{table_id}'
   
    table_obj = bq.Table(table_path)
    table = client.create_table(table_obj)

    dataimport.bq_query_no_return(
        project_id=project_id,
        query=f'SELECT * FROM `{dataset_id}.{table_id}`',
    )

    mock_table.assert_called_with('project.dataset.table')
    mock_client().create_table.assert_called_with(mock_table.return_value)

def test_bq_query_to_dataframe(mocker):
    import google.cloud.bigquery as bq
    import pandas as pd

    mock_table = mocker.patch('google.cloud.bigquery.Table', autospec=True)
    mock_client = mocker.patch('google.cloud.bigquery.Client', autospec=True)

    client = bq.Client()
    project_id = 'project'
    dataset_id = 'dataset'
    table_id = 'table'
    table_path = f'{project_id}.{dataset_id}.{table_id}'
   
    table_obj = bq.Table(table_path)
    table = client.create_table(table_obj)

    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_obj:
        dataset = dataset_obj
        dataset.path = 'tests/unit/resources/dataimport/bq_query_to_dataframe_dataset_output.parquet'
        dataset.metadata = {}
    
    dataimport.bq_query_to_dataframe(
        project_id=project_id,
        query=f'SELECT * FROM `{dataset_id}.{table_id}`',
        dataset=dataset
    )