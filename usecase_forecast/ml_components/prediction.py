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

def prediction(
    bucket_name: str,
    src_run_dt: str,
    sku_name: str,
    freq: str,
    sku_column: str,
    ts_column: str,
    val_column: str,
    metrics_name: str,
    model_params: str,
    forecasting_horizon_days: int,
    interpolated_dataset: Input[Dataset],
    accuracy_board_dataset: Input[Dataset],
    prediction_view: Output[HTML],
):
    from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
    from darts.utils.utils import ModelMode
    from darts.metrics import mape
    from darts.models import (
        Prophet,
        AutoARIMA,
        ExponentialSmoothing,
        FourTheta,
        Theta,
        NBEATSModel,
        NHiTS,
    )
    from darts import TimeSeries

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import base64
    import json
    import io

    sku_df = pd.read_parquet(interpolated_dataset.path)
    sku_df = sku_df.set_index(ts_column, drop=True)

    model_params = json.loads(model_params)

    # Get the best model
    accuracy_board = pd.read_parquet(accuracy_board_dataset.path)
    try:
        idx = accuracy_board.loc[accuracy_board[metrics_name] == accuracy_board[metrics_name].min()].index[0]
    except:
        idx = accuracy_board.index[0]
    allowed_models = [accuracy_board.at[idx,'model']]

    # Return model with a given model name
    def get_model(
        model_name: str='Prophet'
    ):
        # https://unit8co.github.io/darts/generated_api/darts.models.prophet.html
        if model_name == 'Prophet':
            return Prophet(**model_params[model_name])

        # This is actually Holtwinters ES
        # https://unit8co.github.io/darts/generated_api/darts.models.exponential_smoothing.html
        elif model_name == 'ExponentialSmoothing':
            return ExponentialSmoothing(**model_params[model_name])

        # https://unit8co.github.io/darts/generated_api/darts.models.theta.html
        # Need to study more Theta models. Winner for M3 forecast competition.
        elif model_name == 'FourTheta':
            return FourTheta(**model_params[model_name])
        elif model_name == 'Theta':
            return Theta(**model_params[model_name])

        # ARIMA require at least 30 ts data to be used.
        # https://unit8co.github.io/darts/generated_api/darts.models.auto_arima.html
        elif model_name == 'AutoARIMA':
            return AutoARIMA(**model_params[model_name])

        # https://medium.com/@kshavgupta47/n-beats-neural-basis-expansion-analysis-for-interpretable-time-series-forecasting-91e94c830393
        # https://unit8co.github.io/darts/generated_api/darts.models.nbeats.html
        elif model_name == 'nbeats_generic':
            model = NBEATSModel(**model_params[model_name])
            return model

        elif model_name == 'nbeats_interpretable':
            model = NBEATSModel(**model_params[model_name])
            return model

        elif model_name == 'nhits':
            model = NHiTS(**model_params[model_name])
            return model

        else:
            return

    forecast_model = [
        'Prophet',
        'ExponentialSmoothing',
        'FourTheta',
        'Theta',
        'AutoARIMA',
    ]

    forecast_model_V2 = [
        'nbeats_generic',
        'nbeats_interpretable',
        'nhits',
    ]

    # Split data to train & test data
    split_date = sku_df.index[-1]

    sku_df = pd.DataFrame(
        data=sku_df,
        index=pd.date_range(
            start=sku_df.index[0],
            periods=len(sku_df.index) + forecasting_horizon_days,
            freq=freq,
            name=ts_column
        )
    ).reset_index()
    sku_df[[i for i in sku_df.columns if i not in [ts_column, val_column]]] = sku_df[[i for i in sku_df.columns if i not in [ts_column, val_column]]].fillna(method='ffill')

    results = sku_df.set_index(ts_column, drop=False)

    for step_model in allowed_models:
        # forecast_model V1 is for non-neuralnet DART models
        if step_model in forecast_model:
            series = TimeSeries.from_dataframe(sku_df, ts_column, val_column)

            train, val = series.split_after(split_date)

            # Training
            model = get_model(model_name=step_model)
            model.fit(train)
            # Predict
            prediction = model.predict(len(val))

        # forecast_model V2 is for neuralnet DART models (need to to perform scaling)
        elif step_model in forecast_model_V2:
            filler = MissingValuesFiller()
            scaler = Scaler()
            series = scaler.fit_transform(
                filler.transform(
                TimeSeries.from_dataframe(
                    sku_df, 
                    ts_column, 
                    val_column
                )
                )
            )

            train, val = series.split_after(split_date)

            # Training
            model = get_model(model_name=step_model)
            model.fit(
                series=train,
                val_series=val
            )
            # Predict
            prediction = scaler.inverse_transform(model.predict(len(val)))
            series = scaler.inverse_transform(series)

        else:
            raise(Exception(f'Model not present : {step_model}'))

        prediction = prediction.pd_dataframe()

        # Append prediction to result
        prediction.columns = [step_model]
        results[step_model] = prediction[step_model]

    # Start capturing graphs using io
    def plot_to_html(tmpfile):
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        plot_as_html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        return plot_as_html

    # Plot model evaluation graphs
    results = results.rename(columns = {val_column : 'Interpolated'})
    results.plot.line(
        x=ts_column,
        y=['Interpolated'] + allowed_models,
        cmap='Spectral',
        figsize=(10, 10),
        lw=4
    )
    plt.legend()
    plt.ylabel(val_column)
    plt.xlabel(ts_column)
    plt.show()
    
    graph_1 = plot_to_html(io.BytesIO())
    html = f'<p>{sku_column} : {sku_name}</p>' + '<br>' + graph_1 # + '<br>' + graph_2
    with open(prediction_view.path, 'w') as f:
        f.write(html)

    from google.cloud import storage

    ## Save result to local(container)
    pred_save_path = f'prediction_{sku_name}.parquet'
    pred_html_save_path = f'prediction_{sku_name}.html'

    results.to_parquet(pred_save_path, index=False)
    with open(pred_html_save_path, 'w') as f:
        f.write(html)

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name.replace('gs://', ''))

    pred_artifact_path = f'artifacts/prediction/{src_run_dt}/'
    pred_html_artifact_path = f'artifacts/html/prediction/{src_run_dt}/'

    blob = bucket.blob(pred_artifact_path + pred_save_path)
    blob.upload_from_filename(pred_save_path)

    blob = bucket.blob(pred_html_artifact_path + pred_html_save_path)
    blob.upload_from_filename(pred_html_save_path)