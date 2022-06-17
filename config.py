from ml_components.pipelinehelper import restart_model_league
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import pubsub
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
PROJECT_NAME = 'helloworld-classify-vaip' # please use '-' as separator
PIPELINE_NAME = f'{PROJECT_NAME}-{RUNNER}'
BUCKET_NAME = f"gs://{PIPELINE_NAME}"
TRAIN_PIPELINE_ROOT = f'{BUCKET_NAME}/train_pipeline_root/'
PRED_PIPELINE_ROOT = f'{BUCKET_NAME}/pred_pipeline_root/'
MLOPS_TOPIC = 'projects/your_project_id/topics/mlops-endpoint' # dont change
IS_ENDPOINT = True

# DS control
PARAMETER_VALUES = {
    'bucket_name': BUCKET_NAME,
    'project_id': PROJECT_ID,
    'project_name': PROJECT_NAME, # same as usecase_id
    'region': REGION,
    'run_date': '',
    'seed': 0,
    'train_size': 0.8,
    'target_column': 'your_target_column',
    'drop_columns': json.dumps(
        [
            'date'
        ],
    ),
    'auto_balance': 'over_sampling', #''/under_sampling/over_sampling
    'feature_importance_dict_str': json.dumps(
        {
            'V1': 0,
            'V2': 1,
            'V3': 5,
            'V4': 4,
            'V5': 3,
            'V6': 7,
            'V7': 50,
            'V8': 9,
            'V9': 8,
            'V10': 90,
            'V11': 60,
            'V12': 33,
            'V13': 45,
            'V14': 100,
            'V15': 36,
            'V16': 15,
            'V17': 28,
            'V18': 3,
            'V19': 4,
            'V20': 5,
        }
    ),
    'numerical_drift_partition_threshold': 0.5,
    'numerical_importance_partition_threshold': 0.5,
    'categorical_drift_partition_threshold': 0.5,
    'categorical_importance_partition_threshold': 0.5,
    'category_threshold': 5,
    'delta': 1000,
    'model_params': json.dumps(
        {
            'objective': 'binary:logistic',
            'n_estimators': 1000,
            'min_child_weight': 4, 
            'learning_rate': 0.1, 
            'colsample_bytree': 0.6, 
            'max_depth': 10,
            'subsample': 0.8, 
            'gamma': 0.3,
            'booster' : 'gbtree',
            'use_label_encoder' : False
        }        
    ),
    'is_endpoint': IS_ENDPOINT,
    'mlops_topic': MLOPS_TOPIC,
    'runner': RUNNER,
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

    if IS_ENDPOINT:
        # check a pub with topic named 'mlops-endpoint-dotcom' exist or not,
        # if not create that pubsub + topic
        publisher = pubsub.PublisherClient()
        topics = [i.name for i in publisher.list_topics(request={"project": "projects/your_project_id"})]

        if MLOPS_TOPIC not in topics:
            print(f"'{MLOPS_TOPIC}' topic is not exist, starting a new one...")
            publisher.create_topic(request={'name': MLOPS_TOPIC})