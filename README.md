
# mlops-on-vertex-ai-pipelines

Vertex AI Pipeline Base template for MLOps.

# Introduction

You managed to build a good demand forecasting model. or GPT-10. So, what's next?

People tend to think that a Machine Learning project is all just about building models that can do cool things.

![alt text](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/blob/usecase-forecast/docs/resources/images/intro_1.png)

However, it's more that just that. A Machine Learning project is a lifecycle where we
1. Continuously monitor and analyse the project performance from the perspective of business, data quality and pipeline health.
2. Debug or enhance the component of the pipelines.

![alt text](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/blob/usecase-forecast/docs/resources/images/intro_2.png)

This repository provides an end-to-end Machine Learning pipeline that allow us to iteratively develop and monitor the pipeline efficiently.

# Architecture Diagram

Below is the overall achitecture diagram of this MLOps framework, powered by GitLab, Kubeflow Pipeline and Google Cloud Platform.

![alt text](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/blob/usecase-forecast/docs/resources/images/pipeline_deployment_forecast.png)

# Current Features

This MLOps framework is currently

1. Supports custom data import period and queries.
2. Supports custom data preprocessing function.
3. Supports Data Integrity Check (DIC).
    1. Data Quality Check.
    2. Data Drift Check.
4. Supports Automatic Retraining policy (upon failure of DIC).
5. Supports saving data drift, model evaluation, prediction graphs into HTML.
6. Supports Champion and Challenger(multi-model) concept.
7. Store train, validation and prediction snapshot datasets.
8. Store train, validation and prediction persist datasets for dashboarding.
9. Custom Model Registry.
    1. Support model registry by SKU level.
    2. Supports model tagging.
    3. Stores train dataset path.
    4. Stores validation dataset path.
    5. Stores prediction dataset path.

# Getting Started

1. ```git clone git@github.com:aziddddd/mlops-on-vertex-ai-pipelines.git```
2. ```git switch <your_desired_template_branch>```
3. ```git checkout -b <your_working_branch_id>```
4. Input your project configuration in config.py.
5. Input your project_id, region, impersonate_service_account in cicd-pipeline.yml.
6. When developing:
    1. Set **RUNNER='dev'** in config.py.
    2. Do your development.
    3. Test your pipeline by running grand_pipeline_*.ipynb and monitor in VAIP UI.
    4. Perform step 2-3 until you satisfy.
7. After developing:
    1. Set **RUNNER='prod'** in config.py.
8. Once satisfied, push to your branch.
7. Create a Pull Request to merge/deploy your branch to master branch and assign the PR to a reviewer (MLOps Team Lead).
9. The reviewer will merge the PR for you.
