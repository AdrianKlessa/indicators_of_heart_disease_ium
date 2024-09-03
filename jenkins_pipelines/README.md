This directory contains files related to setting up Jenkins pipelines for:
- dataset cleaning (main)
- model training
- model evaluation

Setup with corresponding git branches and Jenkins pipelines is required - this was originally done on a private Jenkins instance.

The pipelines were set up to trigger one after another once the previous pipeline succeeded, with the possibility of manually triggering the train job with custom (non-default) parameters for model training.

An example of the evaluation results is included below.

Notes:
- at one point there was an issue with cleaning up the data. Fixing the issue resulted in greatly improved prediction results and changed the amount of data the model used.
- when starting the project I mixed up test and validation datasets, resulting in incorrect usage of these two terms - the validation set is called test and vice versa.

Example results of the predictions (and their evaluation) for the model are included in this directory.