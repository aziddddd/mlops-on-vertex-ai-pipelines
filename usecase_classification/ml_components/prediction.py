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

def model_predictor(
    target_column: str,
    model_champion_object: Input[Model],
    processed_dataset: Input[Dataset],
    prediction_dataset: Output[Dataset],
):
    import xgboost as xgb
    import pandas as pd

    model = xgb.XGBClassifier()
    model.load_model(model_champion_object.path)

    pred_processed_df = pd.read_parquet(processed_dataset.path)

    x_test = pred_processed_df[[i for i in pred_processed_df.columns if i != target_column]]
    pred_test = pd.DataFrame(model.predict_proba(x_test))

    pred_processed_df['pred_score'] = pred_test[1].tolist()
    pred_processed_df.to_parquet(prediction_dataset.path, index=False)