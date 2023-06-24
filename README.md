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

# Machine Learning Usecase

We currently have templates for the following Machine Learning usecases:
1. [usecase_classification](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/tree/main/usecase_classification)
2. [usecase_forecast](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/tree/main/usecase_forecast)
3. [usecase_recsys](https://github.com/aziddddd/mlops-on-vertex-ai-pipelines/tree/main/usecase_recsys)

# Getting Started

1. ```git clone git@github.com:aziddddd/mlops-on-vertex-ai-pipelines.git```
2. ```git checkout -b <your_working_branch_id>```
3. ```cp -r <your_desired_usecase>/ <your_project_name>/```
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
