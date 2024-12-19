# WE_with_DL

Implementation of the model architecture in [this paper (Wang, H., Wang, Z., Li, Z. & Lee, T. Y. 2020)](https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2020.572195/full).

Follow the followng steps to run the experiment.

> Note: All scripts should be executed from the **parent folder** of BioML project, or you have to modify paths in each files.

## Model Training & 5-fold Cross Validation

> Warning: Before running the experiment, make sure that you have executed `./exp/onehot_kmeans.py` to enable down-sampling method.

1. Adjust parameters in `./exp/WE_with_DL/trainer.py`.

2. Run experiment:
    ```bash
    python ./exp/WE_with_DL/trainer.py
    ```