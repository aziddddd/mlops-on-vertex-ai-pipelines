import kfp.v2.components.component_factory as comp
import google.cloud.aiplatform as aip
from google.cloud import storage
from datetime import datetime
from kfp.v2 import compiler
import pandas as pd
import numpy as np
import pandas_gbq
import random
import os

# import logging
# logging.getLogger("google.cloud.aiplatform.pipeline_jobs").setLevel(logging.WARNING)

def _create_component_from_func(
    func, 
    _component_human_name, 
    *args, 
    **kwargs
):
    # Don't override func.__name__ that causes an execution error.
    # func._component_human_name = f"{func.__name__}_{uuid.uuid4().hex}"
    func._component_human_name =_component_human_name
    return comp.create_component_from_func(func=func, *args, **kwargs)

def func_op(
    func,
    _component_human_name,
    packages_to_install=[],
    base_image='python:3.7',
    cpu_limit='2',
    retry=1,
    memory_limit=None,
    pvolumes={},
    timeout=None,
    *args, 
    **kwargs
):
    # Instantiate an operation right before calling it
    # because the suffix must be dynamically generated.
    op = _create_component_from_func(
        func=func,
        _component_human_name=_component_human_name,
        packages_to_install=[
            'google-cloud-aiplatform',
            'google-cloud-storage',
            'google-cloud-pipeline-components',
            'google-cloud-bigquery[bqstorage,pandas]',
        ] + packages_to_install,
        base_image=base_image,
    )
    task = op(*args, **kwargs)
    task.set_cpu_limit(cpu_limit)
    task.set_retry(retry)

    if pvolumes:
        task.add_pvolumes(pvolumes)

    if memory_limit:
        task.set_memory_limit(memory_limit)

    if timeout:
        task.set_timeout(timeout)
        
    return task

def save_pipeline(
    pipeline,
    pipeline_name,
    bucket_name,
    mode=None,
    
):
    right_now = datetime.now().strftime("%Y-%m-%d-t-%H-%M-%S")

    pipeline_spec_folder = 'pipeline_spec'
    if mode:
        pipeline_spec_folder = f'{mode}_pipeline_spec'
        pipeline_name = f'{pipeline_name}-{mode}'
    job_id = f'{pipeline_name}-{right_now}'
    pipeline_spec_name = f'{job_id}.json'
    push_file_name = f'{pipeline_spec_folder}/{pipeline_spec_name}'
    temp_path = f'/tmp/{pipeline_spec_name}'
    
    # Compile the defined pipeline
    compiler.Compiler().compile(
        pipeline_func=pipeline, 
        package_path=temp_path
    )
    
    # Push pipeline spec to GCS
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name.replace('gs://', ''))
    blob = bucket.blob(push_file_name)
    blob.upload_from_filename(temp_path)
    print(f'{push_file_name} uploaded to GCS.')
    
    # Remove pipeline spec from local
    os.remove(temp_path)
    print(f'{temp_path} removed in local.')
    
    template_path = f'{bucket_name}/{push_file_name}'
    return template_path, job_id, pipeline_name

def run_pipeline(
    project_id:str,
    staging_bucket:str,
    location:str,
    display_name:str,
    template_path:str,
    job_id:str,
    pipeline_root:str,
    parameter_values:dict,
    service_account:str,
    enable_caching:bool=True,
    runner:str='dev',
    use_vaiexp:bool=True,
):

    init_configs = {
        'project': project_id,
        'location': location,
        'staging_bucket': staging_bucket,
    }
    experiment_name = f'{display_name}-experiment'

    if use_vaiexp:
        init_configs['experiment'] = experiment_name

    aip.init(**init_configs)

    # parameter_values['src_run_dt'] = job_id.replace(f'{display_name}-', '')
    parameter_values['job_id'] = job_id

    pipeline_configs = {
        'location': location,
        'display_name': display_name,
        'template_path': template_path,
        'job_id': job_id,
        'pipeline_root': pipeline_root,
        'parameter_values': parameter_values,
    }
    
    if not enable_caching:
        pipeline_configs['enable_caching'] = False

    if parameter_values['runner'] == 'prod':
        try:
            __IPYTHON__
            raise(Exception('Running prod on notebook without permission.'))
        except NameError:
            print('Running prod on Google Cloud Function.')

    job = aip.PipelineJob(**pipeline_configs)

    # Execute the pipeline
    if use_vaiexp:
        job.submit(
            experiment=experiment_name,
            service_account=service_account,
            create_request_timeout=20.0,
        )
    else:
        def _dont_block_until_complete():
            print('_block_until_complete method is turned off for the sake of cloud function completion. Please monitor or control the pipeline through UI.')

        if runner == 'prod':
            job._block_until_complete = _dont_block_until_complete
            sync = False
        elif runner == 'dev':
            sync = True

        job.run(
            service_account=service_account,
            sync=sync
        )

def restart_model_league(
    project_id: str,
    table_id: str,
    location: str='asia-southeast1',
):
    model_league = pd.DataFrame(
        data={
            'src_run_dt': [],
            'tag': [],
            'commit_short_sha': [],
            'user_bq_path': [],
            'catalog_bq_path': [],
            'cg_model_path': [],
            'cg_index_path': [],
            'r_model_path': [],
            'train_bq_path': [],
            'job_id': [],
            'unique_catalog_ids': [],
            'unique_user_ids': [],
            'embedding_dimension': [],
        },
    )
    pandas_gbq.to_gbq(
        model_league,
        destination_table=table_id,
        project_id=project_id,
        location=location,
        if_exists='replace',
    )
    print('model_league has been restarted.')
    print(f'model_league name : {table_id}')
    return model_league