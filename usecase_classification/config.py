from ml_components.pipelinehelper import restart_model_league
import google.cloud.aiplatform as aip
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import pubsub
from icecream import ic
import subprocess
import json
import os

########################################################################################################################
########################################################################################################################
############################################### Parameter Configuration ################################################
########################################################################################################################
########################################################################################################################

# Pipeline configs
SERVICE_ACCOUNT = 'your-service-account@developer.gserviceaccount.com' #'your_service_account'
PROJECT_ID = 'your-project-id' #"your_project_id"
REGION = 'us-central1' #"your_region"
RUNNER = 'prod' #prod/dev
PROJECT_NAME = 'template-classify' # please use '-' as separator
PIPELINE_NAME = f'{PROJECT_NAME}-{RUNNER}'
BUCKET_NAME = f"gs://{PIPELINE_NAME}"
TRAIN_PIPELINE_ROOT = f'{BUCKET_NAME}/train_pipeline_root/'
PRED_PIPELINE_ROOT = f'{BUCKET_NAME}/pred_pipeline_root/'
IS_ENDPOINT = True
USE_VAIEXP = True # ignore this
MLOPS_TOPIC = 'projects/your-project-id/topics/mlops-endpoint-dotcom' # ignore this
MLOPS_SERVICE_ACCOUNT = 'mlops-pipeline-deploy@your-project-id.iam.gserviceaccount.com' # ignore this

if os.path.exists('commit_history.txt'):
    with open('commit_history.txt') as f:
        commit_short_sha = f.read()
else:
    commit_short_sha = ''

# DS control
PARAMETER_VALUES = {
    'bucket_name': BUCKET_NAME,
    'project_id': PROJECT_ID,
    'project_name': PROJECT_NAME, # same as usecase_id
    'region': REGION,
    'run_date': '', # date availability : 2022-06-15 -> 2022-06-18
    'seed': 0,
    'train_size': 0.8,
    'target_column': 'Class',
    'drop_columns': json.dumps(
        [
            'dt'
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
            'V21': 6,
            'V22': 7,
            'V23': 8,
            'V24': 9,
            'V25': 10,
            'V26': 11,
            'V27': 12,
            'V28': 13,
        }
    ),
    'numerical_drift_partition_threshold': 0.01, # X line
    'numerical_importance_partition_threshold': 0.4, # Y line
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
    'commit_short_sha': commit_short_sha,
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
                f"serviceAccount:{SERVICE_ACCOUNT}:roles/storage.admin",
                BUCKET_NAME,
            ]
        )
        subprocess.check_output(
            [
                "gsutil",
                "iam",
                "ch",
                f"serviceAccount:{MLOPS_SERVICE_ACCOUNT}:roles/storage.admin",
                BUCKET_NAME,
            ]
        )
        print("Created bucket, BUCKET_NAME :", BUCKET_NAME)

    if RUNNER not in ['prod', 'dev']:
        raise(Exception('Invalid input for Runner parameter. Please set Runner parameter correctly.'))

    bq_client = bigquery.Client(project=PROJECT_ID)

    # Check MLOps datasets, by right the essential datasets and mlops topic
    # should already be exist when running in prod
    if RUNNER == 'dev':
        datasets = [i.dataset_id for i in bq_client.list_datasets(project=PROJECT_ID)]
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

        if IS_ENDPOINT:
            # check a pub with topic named 'mlops-endpoint-dotcom' exist or not,
            # if not create that pubsub + topic
            publisher = pubsub.PublisherClient()
            topics = [i.name for i in publisher.list_topics(request={"project": "projects/your-project-id"})]

            if MLOPS_TOPIC not in topics:
                print(f"'{MLOPS_TOPIC}' topic is not exist, starting a new one...")
                publisher.create_topic(request={'name': MLOPS_TOPIC})

    # Check model_league table
    TRACKING_DATASET = 'MLOPS_TRACKING'
    tables = [i.table_id for i in bq_client.list_tables(dataset=TRACKING_DATASET)]
    model_league_name = f'{RUNNER}_model_league_{PROJECT_NAME}'
    if not model_league_name in tables:
        print(f"'{model_league_name}' table is not exist, starting a new one...")
        restart_model_league(
            project_id=PROJECT_ID,
            table_id=f'{TRACKING_DATASET}.{model_league_name}',
            location=REGION,
        )

        if USE_VAIEXP:
            print(f"removing old experiment(s)..")
            experiments = [experiment.name for experiment in aip.Experiment.list(project=PROJECT_ID, location=REGION) if experiment.name.startswith(PIPELINE_NAME)]
            for experiment_name in experiments:
                experiment_obj = aip.Experiment(experiment_name)
                experiment_obj.delete()
                print(f"experiment '{experiment_name}' is removed.")

    print('All checks passed.')
