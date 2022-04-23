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
SERVICE_ACCOUNT = 'your_service_account@developer.gserviceaccount.com' #'your_service_account'
PROJECT_ID = 'your-project-id' #"your_project_id"
REGION = 'us-central1' #"your_region"
RUNNER = 'prod' #prod/dev
PIPELINE_NAME = f'mock-grand-pipeline-{RUNNER}'
BUCKET_NAME = f"gs://{PIPELINE_NAME}"
CREATE_BUCKET = False
DATAMODEL_PIPELINE_ROOT = f'{BUCKET_NAME}/datamodel_pipeline_root/'
TRAIN_PIPELINE_ROOT = f'{BUCKET_NAME}/train_pipeline_root/'
PRED_PIPELINE_ROOT = f'{BUCKET_NAME}/pred_pipeline_root/'

# DS control
PARAMETER_VALUES = {
    'bucket_name': BUCKET_NAME,
    'project_id': PROJECT_ID,
    'run_date': '2021-01-04',
    'data_features_dict': json.dumps(
        {
            "culmen_length_mm": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "culmen_depth_mm": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "flipper_length_mm": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "body_mass_g": None,
        }
    ),
    'current_drift_threshold': 10.0,
    'train_drift_threshold': 10.0,
    'pred_drift_threshold': 10.0,
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
    'failure_webhook_config': json.dumps(
        {
        'project_uid': '0123456789012',
        'secret_id': 'your-project-failure-bot',
        'version_id': '1',
        }
    ),
    'progress_webhook_config': json.dumps(
        {
        'project_uid': '0123456789012',
        'secret_id': 'your-project-progress-tracker',
        'version_id': '1',
        }
    ),
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
    elif CREATE_BUCKET:
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
    else:
        print('Bucket not exist, but user did not intend to create one. Refer CREATE_BUCKET variable in config.')

    if RUNNER not in ['prod', 'dev']:
        raise(Exception('Invalid input for Runner parameter. Please set Runner parameter correctly.'))