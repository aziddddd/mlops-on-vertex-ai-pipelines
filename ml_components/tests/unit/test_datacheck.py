from kfp_components import datacheck
from kfp.v2.dsl import (
    Dataset,
    Metrics,
)
from unittest.mock import patch
from unittest import mock
import pytest

@patch('builtins.print')
def test_printing(mocker):
    # The actual test
    datacheck.printing('Test')
    mocker.assert_called_with('Test')

@patch('builtins.print')
def test_data_quality_check(mocker):
    import pandas as pd

    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_obj:
        dataset = dataset_obj
        dataset.path = 'tests/unit/resources/datacheck/data_quality_check.parquet'

    (qc_status,) = datacheck.data_quality_check(
        pred_month='test',
        dataset=dataset,
    )
    assert(qc_status), 'Failed data quality check function, expected pass'

@patch('builtins.print')
def test_get_quality_check_result(mocker):
    (alert_msg,) = datacheck.get_quality_check_result(
        train_quality_check=True,
        pred_quality_check=True
    )
    mocker.assert_called_with('All data integrity checks passed.')
    assert(not alert_msg), "alert_msg should be empty string '' since all data qc check passed."

    (alert_msg,) = datacheck.get_quality_check_result(
        train_quality_check=True,
        pred_quality_check=False
    )
    assert(alert_msg), "alert_msg should be flag out the failed data quality check."

@patch('builtins.print')
def test_drift_check(mocker):
    with patch(
        target='kfp.v2.dsl.Metrics', 
    ) as feature_drifts_obj:
        feature_drifts = feature_drifts_obj

    # Case 1 : dataset q empty
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq_empty.parquet'

    features_dict_str = '{"test_column": null}'

    (drift_check_message,) = datacheck.kld_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        drift_threshold=1.0,
        mode='test',
        feature_drifts=feature_drifts,
    )
    mocker.assert_called_with('No dataset q, possibility of cold start, passing through.')
    assert(drift_check_message), 'Failed drift check function, expected pass'

    # Case 2 : dataset q no empty
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq.parquet'

    features_dict_str = '{"test_column": null}'

    (drift_check_message,) = datacheck.kld_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        drift_threshold=1.0,
        mode='test',
        feature_drifts=feature_drifts,
    )
    assert(drift_check_message), 'Failed drift check function, expected pass'

    # Case 3 : features_dict_str not None
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq.parquet'

    features_dict_str = '{"test_column": [1, 2]}'

    (drift_check_message,) = datacheck.kld_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        drift_threshold=1.0,
        mode='test',
        feature_drifts=feature_drifts,
    )
    assert(drift_check_message), 'Failed drift check function, expected pass'

    # Case 4 : if global_drift > drift_threshold
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq.parquet'

    features_dict_str = '{"test_column": [1, 2]}'

    (drift_check_message,) = datacheck.kld_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        drift_threshold=-0.0000001, #because mock global_drift = 0
        mode='test',
        feature_drifts=feature_drifts,
    )
    assert(not(drift_check_message)), 'Passed drift check function, expected fail'

@patch('builtins.print')
def test_hd_drift_check(mocker):
    feature_importance_dict_str = '{}'
    with patch(
        target='kfp.v2.dsl.Metrics', 
    ) as feature_drifts_obj:
        feature_drifts = feature_drifts_obj

    # Case 1 : dataset q empty
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq_empty.parquet'

    features_dict_str = '{"test_column": null}'

    (drift_check_message,) = datacheck.hd_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        feature_importance_dict_str=feature_importance_dict_str,
        drift_threshold=1.0,
        mode='test',
        feature_drifts=feature_drifts,
    )
    mocker.assert_called_with('No dataset q, possibility of cold start, passing through.')
    assert(drift_check_message), 'Failed drift check function, expected pass'

    # Case 2 : dataset q no empty
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq.parquet'

    features_dict_str = '{"test_column": null}'

    (drift_check_message,) = datacheck.hd_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        feature_importance_dict_str=feature_importance_dict_str,
        drift_threshold=1.0,
        mode='test',
        feature_drifts=feature_drifts,
    )
    assert(drift_check_message), 'Failed drift check function, expected pass'

    # Case 3 : features_dict_str not None
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq.parquet'

    features_dict_str = '{"test_column": [1, 2]}'

    (drift_check_message,) = datacheck.hd_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        feature_importance_dict_str=feature_importance_dict_str,
        drift_threshold=1.0,
        mode='test',
        feature_drifts=feature_drifts,
    )
    assert(drift_check_message), 'Failed drift check function, expected pass'

    # Case 4 : if global_drift > drift_threshold
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq.parquet'

    features_dict_str = '{"test_column": [1, 2]}'

    (drift_check_message,) = datacheck.hd_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        feature_importance_dict_str=feature_importance_dict_str,
        drift_threshold=-0.0000001, #because mock global_drift = 0
        mode='test',
        feature_drifts=feature_drifts,
    )
    assert(not(drift_check_message)), 'Passed drift check function, expected fail'

    # Case 5 : feature importance dict values more than 1.0
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq.parquet'

    feature_importance_dict_str = '{"test_column": 1.5}'
    features_dict_str = '{"test_column": [1, 2]}'

    with pytest.raises(Exception) as execinfo:
        (drift_check_message,) = datacheck.hd_drift_check(
            dataset_p=dataset_pq,
            dataset_q=dataset_pq,
            features_dict_str=features_dict_str,
            feature_importance_dict_str=feature_importance_dict_str,
            drift_threshold=-0.0000001, #because mock global_drift = 0
            mode='test',
            feature_drifts=feature_drifts,
        )
    assert(str(execinfo.value) == 'Boosted value of feature importance dict must be equal to 1'), 'Incorrect summation of boosted feature importance error message not raised properly'

    # Case 6 : only 1 feature mentioned in feature importance dict, another 1 is not
    with patch(
        target='kfp.v2.dsl.Dataset', 
    ) as dataset_pq_obj:
        dataset_pq = dataset_pq_obj
        dataset_pq.path = 'tests/unit/resources/datacheck/drift_check_dataset_pq_two_cols.parquet'

    feature_importance_dict_str = '{"test_column": 1.0}'
    features_dict_str = '{"test_column": [1, 2], "another_test_column": [1, 2]}'

    (drift_check_message,) = datacheck.hd_drift_check(
        dataset_p=dataset_pq,
        dataset_q=dataset_pq,
        features_dict_str=features_dict_str,
        feature_importance_dict_str=feature_importance_dict_str,
        drift_threshold=1.0,
        mode='test',
        feature_drifts=feature_drifts,
    )
    assert(drift_check_message), 'Failed drift check function, expected pass'