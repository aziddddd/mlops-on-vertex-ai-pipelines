# from kfp_components import prediction
# from kfp.v2.dsl import (
#     Dataset,
#     Model,
#     Metrics,
#     HTML,
# )
# from unittest.mock import patch
# from unittest import mock
# import pytest

# def test_get_decision(mocker):
#     cases = [
#         # Case 0 : All drift check passed
#         {
#             'best_model': 'champion',
#             'adhoc_model_selection': '',
#             'current_drift_check': True,
#             'train_drift_check': True,
#             'pred_drift_check': True,
#             'model_drift_check': True,
#         },
#         # Case 1 : DS provide adhoc_model_selection
#         {
#             'best_model': 'champion',
#             'adhoc_model_selection': 'challenger',
#             'current_drift_check': True,
#             'train_drift_check': True,
#             'pred_drift_check': True,
#             'model_drift_check': True,
#         },
#         # Case 2 : One or more of the drift check fails.
#         {
#             'best_model': 'champion',
#             'adhoc_model_selection': '',
#             'current_drift_check': False,
#             'train_drift_check': False,
#             'pred_drift_check': True,
#             'model_drift_check': True,
#         },
#     ]

#     (choose_model, alert_msg) = prediction.get_decision(**cases[0])
#     assert(not alert_msg),'case 0: alert_msg is retrieved when it should not be.'

#     (choose_model, alert_msg) = prediction.get_decision(**cases[1])
#     assert(not alert_msg),'case 1: alert_msg is retrieved when it should not be.'

#     (choose_model, alert_msg) = prediction.get_decision(**cases[2])
#     assert(alert_msg),'case 2: alert_msg is not retrieved when it should be.'

# def test_model_predictor(mocker):
#     with patch(
#         target='kfp.v2.dsl.Model', 
#     ) as model_object_mock:
#         model_object = model_object_mock
#         model_object.path = 'tests/unit/resources/modelling/model_evaluator_model_object'

#     with patch(
#         target='kfp.v2.dsl.Model', 
#     ) as model_champion_object_mock:
#         model_champion_object = model_champion_object_mock
#         model_champion_object.path = 'tests/unit/resources/modelling/model_evaluator_model_object'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as processed_dataset_obj:
#         processed_dataset = processed_dataset_obj
#         processed_dataset.path = 'tests/unit/resources/prediction/model_predictor_processed_dataset_input.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as output_check_dataset_obj:
#         output_check_dataset = output_check_dataset_obj
#         output_check_dataset.path = 'tests/unit/resources/prediction/model_predictor_output_check_dataset_output.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as prediction_dataset_obj:
#         prediction_dataset = prediction_dataset_obj
#         prediction_dataset.path = 'tests/unit/resources/prediction/model_predictor_prediction_dataset_output.parquet'

#     # Case 1 : Alert message exist.
#     choose_model = 'champion'
#     alert_msg = 'Drift check(s) failed : current_drift_check'
#     with pytest.raises(Exception) as execinfo:
#         (oc_status, ) = prediction.model_predictor(
#             choose_model=choose_model,
#             alert_msg=alert_msg,
#             model_object=model_object,
#             model_champion_object=model_champion_object,
#             processed_dataset=processed_dataset,
#             output_check_dataset=output_check_dataset,
#             prediction_dataset=prediction_dataset,
#         )
#     assert(str(execinfo.value).startswith('Drift check(s) failed : ')), 'Drift check(s) error message not raised properly'

#     # Case 2 : Alert message not exist, pipelien continues
#     for chs_model in ['champion', 'challenger']:
#         alert_msg = ''
#         (oc_status, ) = prediction.model_predictor(
#             choose_model=chs_model,
#             alert_msg=alert_msg,
#             model_object=model_object,
#             model_champion_object=model_champion_object,
#             processed_dataset=processed_dataset,
#             output_check_dataset=output_check_dataset,
#             prediction_dataset=prediction_dataset,
#         )
#         assert(oc_status in ['success', 'failed']), f'Unknown oc_status : {oc_status}'