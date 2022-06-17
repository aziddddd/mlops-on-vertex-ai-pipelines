from ml_components.pipelinehelper import restart_model_league
from google.cloud import bigquery
from google.cloud import storage
from icecream import ic
import subprocess
import json

########################################################################################################################
########################################################################################################################
############################################### Parameter Configuration ################################################
########################################################################################################################
########################################################################################################################

# Pipeline configs
SERVICE_ACCOUNT = 'your_service_account' #'your_service_account'
PROJECT_ID = 'your_project_id' #"your_project_id"
REGION = 'your_region' #"your_region"
RUNNER = 'prod' #prod/dev
PROJECT_NAME = 'helloworld-forecast-vaip' # please use '-' as separator
PIPELINE_NAME = f'{PROJECT_NAME}-{RUNNER}'
BUCKET_NAME = f"gs://{PIPELINE_NAME}"
E2E_PIPELINE_ROOT = f'{BUCKET_NAME}/pipeline_root/'

# DS control
PARAMETER_VALUES = {
    'bucket_name': BUCKET_NAME,
    'project_id': PROJECT_ID,
    'project_name': PROJECT_NAME,
    'region': REGION,
    'run_date': '',
    'train_size': 0.8,
    'training_months': 12,
    'forecasting_horizon_days': 6*30,
    'sku_column': 'table_name',
    'ts_column': 'snapshot_date',
    'val_column': 'row_count',
    'metrics_name': 'MAPE', # 'wMAPE'
    'freq': 'D',
    'allowed_models': json.dumps(
        [
            'Prophet',
            'ExponentialSmoothing',
            'nhits',
            'nbeats_interpretable',
            # 'nbeats_generic',
            # 'AutoARIMA',
            # 'FourTheta',
            # 'Theta'
        ]
    ),
    'feature_importance_dict_str': json.dumps(
        {
            # 'your_val_column': 1,
        }
    ),
    'numerical_drift_partition_threshold': 0.00001,
    'numerical_importance_partition_threshold': 0.5,
    'categorical_drift_partition_threshold': 0.5,
    'categorical_importance_partition_threshold': 0.5,
    'category_threshold': 1,
    'delta': 1000,
    'model_params': json.dumps(
        {
            # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html
            'Prophet': {
                # 'seasonality_mode': 'additive',
                # 'yearly_seasonality': False,
                # 'weekly_seasonality': False,
                # 'daily_seasonality': True,
            },
            'ExponentialSmoothing': {},
            'FourTheta': {},
            'Theta': {},
            #https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html
            'AutoARIMA': {
                'start_p': 1, 'start_q': 1, 
                'max_p': 8, 'max_q': 8,
                'start_P': 0, 'start_Q': 0, 
                'max_P': 8, 'max_Q': 8,
                'm': 12, 
                'seasonal': True,
                'trace': True,
                'd': 1, 'D': 1,
                'error_actuon': 'warn', 
                'suppress_warnings': True,
                'stepwise': True, 
                'random_state': 20, 
                'n_fits': 30
            },
            # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html
            'nbeats_generic': {
                'generic_architecture': True,
                'num_stacks': 10,
                'num_blocks': 1,
                'num_layers': 4,
                'layer_widths': 512,
                'input_chunk_length': 30, # learn 30 days back
                'output_chunk_length': 7, # guess 7 days infront
                'n_epochs': 100,
                'nr_epochs_val_period': 1,
                'batch_size': 16,
            },
            'nbeats_interpretable': {
                'generic_architecture': False,
                'num_blocks': 3,
                'num_layers': 4,
                'layer_widths': 512,
                'input_chunk_length': 30, # learn 30 days back
                'output_chunk_length': 7, # guess 7 days infront
                'n_epochs': 100,
                'nr_epochs_val_period': 1,
                'batch_size': 16,
            },
            # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html
            'nhits': {
                'input_chunk_length': 14, # learn 14 days back
                'output_chunk_length': 7, # guess 7 days infront
                'num_stacks': 10, 
                'num_blocks': 1, 
                'num_layers': 4, 
                'layer_widths': 512,
                'dropout': 0.5,
            },
        } 
    ),
    'runner': RUNNER,
    'tracking_cutoff': 7,
}

########################################################################################################################
########################################################################################################################
#################################################### Parameter Check ###################################################
########################################################################################################################
########################################################################################################################

def parameter_checks(): # pragma: no cover
    PIPELINE_CONFIGS = [
        SERVICE_ACCOUNT,
        PROJECT_ID,
        REGION,
        PIPELINE_NAME
    ]
    if not all(PIPELINE_CONFIGS):
        raise(Exception('SERVICE_ACCOUNT/PROJECT_ID/REGION/PIPELINE_NAME are not set. Please set all required pipeline configs.'))

    ic(SERVICE_ACCOUNT)
    ic(PROJECT_ID)
    ic(REGION)
    ic(PIPELINE_NAME)

    client = storage.Client()    
    bucket = client.bucket(BUCKET_NAME.replace("gs://", ""))
    if bucket.exists():
        print("BUCKET_NAME exists :", BUCKET_NAME)
    else:
        print("BUCKET_NAME not exists, creating a new one...")
        subprocess.check_output(
            [
                "gsutil",
                "mb",
                "-l",
                REGION,
                BUCKET_NAME,
            ]
        )
        subprocess.check_output(
            [
                "gsutil",
                "iam",
                "ch",
                f"serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectAdmin",
                BUCKET_NAME,
            ]
        )
        print("Created bucket, BUCKET_NAME :", BUCKET_NAME)

    if RUNNER not in ['prod', 'dev']:
        raise(Exception('Invalid input for Runner parameter. Please set Runner parameter correctly.'))

    client = bigquery.Client()

    # Check MLOps datasets
    datasets = [i.dataset_id for i in client.list_datasets()]
    mlops_datasets = [
        'MLOPS_PREDICTION_DATASET',
        'MLOPS_TRACKING',
        'MLOPS_TRAIN_DATASET',
        'MLOPS_VAL_DATASET',
    ]

    for mlops_dataset in mlops_datasets:
        if mlops_dataset not in datasets:
            dataset_id = f"{PROJECT_ID}:{mlops_dataset}"
            print(f"MLOps dataset '{dataset_id}' is not exists, creating a new one..")
            subprocess.check_output(
                [
                    "bq",
                    f"--location={REGION}",
                    "mk",
                    dataset_id,
                ]
            )

    # Check model_league table
    TRACKING_DATASET = 'MLOPS_TRACKING'
    tables = [i.table_id for i in client.list_tables(TRACKING_DATASET)]
    model_league_name = f'{RUNNER}_model_league_{PROJECT_NAME}'
    if not model_league_name in tables:
        print(f"'{model_league_name}' table is not exist, starting a new one...")
        restart_model_league(
            project_id=PROJECT_ID,
            table_id=f'{TRACKING_DATASET}.{model_league_name}',
            location=REGION,
        )
