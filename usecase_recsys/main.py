def pipeline_runner(event, context):
    import os
    os.system('python3 grand_pipeline_train.py')
