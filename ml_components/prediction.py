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
    model_champion_object: Input[Model],
    processed_dataset: Input[Dataset],
    output_check_dataset: Output[Dataset],
    prediction_dataset: Output[Dataset],
) -> NamedTuple(
    "Outputs",
    [
        ("oc_status", str),
    ],
):
    import xgboost as xgb
    import pandas as pd
    
    model = xgb.XGBClassifier()
    model.load_model(model_champion_object.path)

    pred_processed_df = pd.read_parquet(processed_dataset.path)

    x_test = pred_processed_df[[i for i in pred_processed_df.columns if i != 'species']]

    pred_test = pd.DataFrame(model.predict_proba(x_test))

    pred_processed_df['pred_score'] = pred_test[1].tolist()

    # Output Check
    label_col = "pred_score"
    sliced_test = pred_processed_df[["culmen_length_mm", label_col]]

    output_check = sliced_test.groupby(
        pd.cut(
            sliced_test[label_col], 
            [0, 0.5, 1.0],
        )
    ).count()
    output_check['%'] = output_check[label_col]/output_check[label_col].sum()*100
    output_check.to_parquet(output_check_dataset.path, index=False)

    def interval_percentage_check(
        lower_boundary: float,
        upper_boundary: float,
        lower_interval: float,
        upper_interval: float,
    ):
        if (lower_boundary < output_check.loc[pd.Interval(lower_interval, upper_interval, closed='right'), '%'] < upper_boundary):
            print('Abnormal percentage for bin {}-{} : {}'.format(lower_interval, upper_interval, output_check.loc[pd.Interval(lower_interval, upper_interval, closed='right'), '%']))
            return False
        else:
            return True
    
    interval_percentage = [
        {'lower_boundary': 20.0, 'upper_boundary': 25.0, 'lower_interval': 0.0, 'upper_interval': 0.5},
        {'lower_boundary': 20.0, 'upper_boundary': 24.0, 'lower_interval': 0.5, 'upper_interval': 1.0},
    ]

    # Check percentage-interval
    check_1 = [interval_percentage_check(**input_check) for input_check in interval_percentage]

    all_check = check_1
    if all(i for i in all_check):
        oc_status = 'success'
    else:
        oc_status = 'failed'

    pred_processed_df.to_parquet(prediction_dataset.path, index=False)

    return (
        oc_status,
    )