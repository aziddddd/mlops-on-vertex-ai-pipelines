# from kfp_components import modelling
# from kfp.v2.dsl import (
#     Dataset,
#     Model,
#     Metrics,
#     HTML,
# )
# from unittest.mock import patch
# from unittest import mock
# import pytest

# def test_model_trainer(mocker):
#     import json
    
#     model_params = json.dumps(
#         {
#             'objective': 'binary:logistic',
#             'n_estimators': 10,
#             'min_child_weight': 4, 
#             'learning_rate': 0.1, 
#             'colsample_bytree': 0.6, 
#             'max_depth': 10,
#             'subsample': 0.8, 
#             'gamma': 0.3,
#             'booster' : 'gbtree',
#             'use_label_encoder' : False
#         }        
#     )

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as x_train_dataset_obj:
#         x_train_dataset = x_train_dataset_obj
#         x_train_dataset.path = 'tests/unit/resources/modelling/model_trainer_x_train_dataset_input.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as x_val_dataset_obj:
#         x_val_dataset = x_val_dataset_obj
#         x_val_dataset.path = 'tests/unit/resources/modelling/model_trainer_x_val_dataset_input.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as y_train_datasetobj:
#         y_train_dataset = y_train_datasetobj
#         y_train_dataset.path = 'tests/unit/resources/modelling/model_trainer_y_train_dataset_input.parquet'

#     with patch(
#         target='kfp.v2.dsl.Dataset', 
#     ) as y_val_dataset_obj:
#         y_val_dataset = y_val_dataset_obj
#         y_val_dataset.path = 'tests/unit/resources/modelling/model_trainer_y_val_dataset_input.parquet'

#     with patch(
#         target='kfp.v2.dsl.Model', 
#     ) as model_object_mock:
#         model_object = model_object_mock
#         model_object.path = 'tests/unit/resources/modelling/model_trainer_model_object'

#     modelling.model_trainer(
#         model_params=model_params,
#         x_train_dataset=x_train_dataset,
#         x_val_dataset=x_val_dataset,
#         y_train_dataset=y_train_dataset,
#         y_val_dataset=y_val_dataset,
#         model_object=model_object,
#     )

# def test_model_evaluator(mocker):
#     for is_cold_start in [True, False]:
#         with patch(
#             target='kfp.v2.dsl.Model', 
#         ) as model_object_mock:
#             model_object = model_object_mock
#             model_object.path = 'tests/unit/resources/modelling/model_evaluator_model_object'

#         with patch(
#             target='kfp.v2.dsl.Model', 
#         ) as model_champion_object_mock:
#             model_champion_object = model_champion_object_mock
#             model_champion_object.path = 'tests/unit/resources/modelling/model_evaluator_model_object'

#         with patch(
#             target='kfp.v2.dsl.Dataset', 
#         ) as x_train_dataset_obj:
#             x_train_dataset = x_train_dataset_obj
#             x_train_dataset.path = 'tests/unit/resources/modelling/model_evaluator_x_train_dataset_input.parquet'

#         with patch(
#             target='kfp.v2.dsl.Dataset', 
#         ) as x_val_dataset_obj:
#             x_val_dataset = x_val_dataset_obj
#             x_val_dataset.path = 'tests/unit/resources/modelling/model_evaluator_x_val_dataset_input.parquet'

#         with patch(
#             target='kfp.v2.dsl.Dataset', 
#         ) as y_val_dataset_obj:
#             y_val_dataset = y_val_dataset_obj
#             y_val_dataset.path = 'tests/unit/resources/modelling/model_evaluator_y_val_dataset_input.parquet'

#         with patch(
#             target='kfp.v2.dsl.HTML', 
#         ) as feature_importance_view_object:
#             feature_importance_view = feature_importance_view_object
#             feature_importance_view.path = 'tests/unit/resources/modelling/model_evaluator_feature_importance_view'

#         with patch(
#             target='kfp.v2.dsl.Metrics', 
#         ) as metrics_object_mock:
#             metrics_object = metrics_object_mock

#         with patch(
#             target='kfp.v2.dsl.Dataset', 
#         ) as model_eval_score_object_mock:
#             model_eval_score_object = model_eval_score_object_mock
#             model_eval_score_object.path = 'tests/unit/resources/modelling/model_evaluator_model_eval_score_object_output.parquet'

#         with patch(
#             target='kfp.v2.dsl.Dataset', 
#         ) as model_champion_eval_score_object_mock:
#             model_champion_eval_score_object = model_champion_eval_score_object_mock
#             model_champion_eval_score_object.path = 'tests/unit/resources/modelling/model_evaluator_model_champion_eval_score_object_output.parquet'

#         modelling.model_evaluator(
#             is_cold_start=is_cold_start,
#             model_object=model_object,
#             model_champion_object=model_champion_object,
#             x_train_dataset=x_train_dataset,
#             x_val_dataset=x_val_dataset,
#             y_val_dataset=y_val_dataset,
#             feature_importance_view=feature_importance_view,
#             metrics_object=metrics_object,
#             model_eval_score_object=model_eval_score_object,
#             model_champion_eval_score_object=model_champion_eval_score_object,
#         )