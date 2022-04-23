
# mlops-on-vertex-ai-pipelines

Vertex AI Pipeline Base template for MLOps.

# Installation

To use this locally:

1. Fork this repository.
2. Never edit directly on master branch so create a new branch.
3. Input your project configuration in config.py
4. When developing:
    1. Set **RUNNER='dev'** in config.py.
    2. Do your development.
    3. Test your pipeline by running grand_pipeline.ipynb and monitor in VAIP UI.
    4. Perform step 2-3 until you satisfy.

5. After developing:
    1. Set **RUNNER='prod'** in config.py.
    2. Add/Improve unit test to accomodate your edit.
    3. Before even pushing to your branch, run the following commands to **perform unit testing locally**.
    ```
    cd kfp_components;
    tests/run
    ```

By unit testing. Refer quote below:
> “Tests are stories we tell the next generation of programmers on a project.”
> ― Roy Osherove, The Art of Unit Testing

6. Once unit test is succeed, push to your branch.
7. Create a Pull Request to merge your branch to master branch and assign the PR to a reviewer (Senior/Lead MLOps).
8. The reviewer will merge the PR for you.

# Important Notes

First time run require model league to be created. Use restart_model_league function in ml_components/pipelinehelper.py as a kickstart to do this.