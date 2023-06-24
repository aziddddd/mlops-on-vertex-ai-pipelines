# mlops-on-vertex-ai-pipelines

Vertex AI Pipeline Base template for MLOps.

# Introduction

You managed to build a good demand forecasting model. or GPT-10. So, what's next?

People tend to think that a Machine Learning project is all just about building models that can do cool things.

![alt text](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/blob/main/docs/resources/images/intro_1.png)

However, it's more that just that. A Machine Learning project is a lifecycle where we
1. Continuously monitor and analyse the project performance from the perspective of business, data quality and pipeline health.
2. Debug or enhance the component of the pipelines.

![alt text](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/blob/main/docs/resources/images/intro_2.png)

This repository provides an end-to-end Machine Learning pipeline that allow us to iteratively develop and monitor the pipeline efficiently.

# Architecture Diagram

Below is the overall achitecture diagram of this MLOps framework, powered by Github, Kubeflow Pipeline and Google Cloud Platform.

![alt text](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/blob/main/docs/resources/images/pipeline_deployment_recsys.png)c

# Current Features

This MLOps framework is currently

1. Supports datamodel generation (optional).
2. Supports custom data import period and queries.
3. Supports custom data preprocessing function.
4. Supports Data Integrity Check (DIC).
    1. Data Quality Check.
    2. Data Drift Check.
5. Supports Automatic Retraining policy (upon failure of DIC).
6. Supports saving model evaluation graphs into HTML.
7. Supports Champion and Challenger concept.
8. Stores train and prediction dataset.
9. Custom Model Registry.
    1. Supports model tagging.
    2. Stores model path.
    3. Stores train dataset path.
    4. Stores prediction dataset paths.
    5. Stores Vertex AI Pipelines job IDs.

# Getting Started

1. ```git clone git@github.com:aziddddd/mlops-on-vertex-ai-pipelines.git```
2. ```git checkout -b <your_working_branch_id>```
3. ```cp -r usecase_recsys/ <your_project_name>/```
4. ```cd <your_project_name>/```
5. Input your project configuration in config.py.
6. Input your project_id, region, impersonate_service_account in cicd-pipeline.yml.
7. When developing:
    1. Set **RUNNER='dev'** in config.py.
    2. Do your development.
    3. Test your pipeline by running grand_pipeline_*.ipynb and monitor in VAIP UI.
    4. Perform step 2-3 until you satisfy.
8. After developing:
    1. Set **RUNNER='prod'** in config.py.
9. Once satisfied, push to your branch.
10. Create a Pull Request to merge/deploy your branch to master branch and assign the PR to a reviewer (MLOps Team Lead).
11. The reviewer will merge the PR for you.
