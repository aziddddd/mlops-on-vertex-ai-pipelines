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

def modelling(
    drift_status: bool,
    project_id: str,
    project_name: str,
    runner:str,
    bucket_name: str,
    src_run_dt: str,
    sku_name: str,
    train_size: float,
    sku_column: str,
    ts_column: str,
    val_column: str,
    allowed_models : str,
    model_params: str,
    previous_train_dataset: Input[Dataset],
    previous_val_dataset: Input[Dataset],
    interpolated_dataset: Input[Dataset],
    accuracy_board_dataset: Output[Dataset],
    model_evaluation_view: Output[HTML],
):
    if drift_status:
        import pandas as pd
        import pandas_gbq

        accuracy_board_path = f'MLOPS_TRACKING.{runner}_accuracy_board_{project_name}'
        accuracy_board_query = f"""
        SELECT * FROM `{accuracy_board_path}`
        WHERE {sku_column} = '{sku_name}'
        """
        accuracy_board = pandas_gbq.read_gbq(
            query=accuracy_board_query, 
            project_id=project_id,
            use_bqstorage_api=True,
        )

        accuracy_board.to_parquet(accuracy_board_dataset.path, index=False)

        train = pd.read_parquet(previous_train_dataset.path)
        val = pd.read_parquet(previous_val_dataset.path)

        from google.cloud import storage

        # For post-process collection later
        ## Save result to local(container)
        board_save_path = f'board_{sku_name}.parquet'
        train_save_path = f'train_{sku_name}.parquet'
        val_save_path = f'val_{sku_name}.parquet'

        accuracy_board.to_parquet(board_save_path, index=False)
        train.to_parquet(train_save_path, index=False)
        val.to_parquet(val_save_path, index=False)

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name.replace('gs://', ''))

        board_artifact_path = f'artifacts/board/{src_run_dt}/'
        train_artifact_path = f'artifacts/train/{src_run_dt}/'
        val_artifact_path = f'artifacts/val/{src_run_dt}/'
        modelling_html_artifact_path = f'artifacts/html/modelling/{src_run_dt}/'

        blob = bucket.blob(board_artifact_path + board_save_path)
        blob.upload_from_filename(board_save_path)

        blob = bucket.blob(train_artifact_path + train_save_path)
        blob.upload_from_filename(train_save_path)

        blob = bucket.blob(val_artifact_path + val_save_path)
        blob.upload_from_filename(val_save_path)

    else:
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
        allowed_models = json.loads(allowed_models)
        model_params = json.loads(model_params)

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

        # Function to calculate weighted MAPE
        def wmape(
            actual, 
            forecast
        ):
            actual, forecast = np.array(actual), np.array(forecast)
            # difference
            diff = np.abs(actual-forecast)
            # weightage for each
            weightage = np.abs(actual)/(np.abs(actual)).sum()
            # weighted difference
            weighted_difference = weightage*diff
            # weighted actual
            weighted_actual= weightage*np.abs(actual)
            # wmape
            wmape = weighted_difference.sum() / weighted_actual.sum()
            return (wmape *100)

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

        accuracy_table = {
            sku_column : [],
            'model' : [],
            'MAPE': [],
            'wMAPE': []
        }

        # Split data to train & test data
        split_date = sku_df.at[
            int(train_size*len(sku_df))-1,
            ts_column
        ]

        results = sku_df.set_index(ts_column, drop=False)

        for step_model in allowed_models:
            try:
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
            except ValueError as err:
                print(f'ValueError : {err}')
                print(f'Skipping this model [{step_model}]...')
                continue

            try:
                MAPE = round(mape(prediction, series), 2)
            except:
                MAPE = np.nan

            # Calculate MAPE, wMAPE for the corresponding model
            prediction_wmape = prediction.pd_series()
            series_wmape = series.pd_series()[prediction_wmape.index]
            WMAPE = round(wmape(series_wmape, prediction_wmape), 2)

            prediction = prediction.pd_dataframe()

            # Add the metrics calculation to table
            accuracy_table[sku_column].append(sku_name)
            accuracy_table['model'].append(step_model)
            accuracy_table['MAPE'].append(MAPE)
            accuracy_table['wMAPE'].append(WMAPE)

            # Append prediction to result
            prediction.columns = [step_model]
            results[step_model] = prediction[step_model]

        accuracy_board = pd.DataFrame(accuracy_table)
        train = results[results['snapshot_date']<=split_date].dropna(how='all', axis=1)
        val = results[results['snapshot_date']>split_date]

        # Start capturing graphs using io
        def plot_to_html(tmpfile):
            plt.savefig(tmpfile, format='png')
            encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
            plot_as_html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
            return plot_as_html

        # Plot model evaluation graphs
        results = results.rename(columns = {val_column : 'Interpolated'})
        plot_models = [i for i in allowed_models if i in results.columns]

        results.plot.line(
            x=ts_column,
            y=['Interpolated'] + plot_models,
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
        with open(model_evaluation_view.path, 'w') as f:
            f.write(html)

        accuracy_board.to_parquet(accuracy_board_dataset.path, index=False)

        from google.cloud import storage

        # For post-process collection later
        ## Save result to local(container)
        board_save_path = f'board_{sku_name}.parquet'
        train_save_path = f'train_{sku_name}.parquet'
        val_save_path = f'val_{sku_name}.parquet'
        modelling_html_save_path = f'modelling_{sku_name}.html'

        accuracy_board.to_parquet(board_save_path, index=False)
        train.to_parquet(train_save_path, index=False)
        val.to_parquet(val_save_path, index=False)
        with open(modelling_html_save_path, 'w') as f:
            f.write(html)

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name.replace('gs://', ''))

        board_artifact_path = f'artifacts/board/{src_run_dt}/'
        train_artifact_path = f'artifacts/train/{src_run_dt}/'
        val_artifact_path = f'artifacts/val/{src_run_dt}/'
        modelling_html_artifact_path = f'artifacts/html/modelling/{src_run_dt}/'

        blob = bucket.blob(board_artifact_path + board_save_path)
        blob.upload_from_filename(board_save_path)

        blob = bucket.blob(train_artifact_path + train_save_path)
        blob.upload_from_filename(train_save_path)

        blob = bucket.blob(val_artifact_path + val_save_path)
        blob.upload_from_filename(val_save_path)

        blob = bucket.blob(modelling_html_artifact_path + modelling_html_save_path)
        blob.upload_from_filename(modelling_html_save_path)