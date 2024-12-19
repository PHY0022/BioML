# DeepAFP

Implementation of the model architecture in [this paper (Yao, L. et al. 2023)](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4758?msockid=08ab38e43da369ce130e2cc13c3768bb).

Follow the followng steps to run the experiment.

> Note: All scripts should be executed from the parent folder of BioML project, or you have to modify paths in each files.

## Features preparation
1. AA features: 
    ```bash
    python ./exp/DeepAFP/aa_features.py
    ```

2. TAPE ecoding:    
    ```bash
    python ./exp/DeepAFP/tape_features.py
    ```

3. ANKH encoding:
    ```bash
    python ./exp/DeepAFP/ankh_features.py
    ```

4. ProTrans encoding:
    ```bash
    python ./exp/DeepAFP/protrans_features.py
    ```

5. ESM2 encoding:
    ```bash
    python ./exp/DeepAFP/esm2_features.py
    ```

After above commands, features will be stored under folder `./exp/DeepAFP/data/`.

## Model Training & 5-fold Cross Validation

> Warning: Before running the experiment, make sure that you have executed `./exp/onehot_kmeans.py` to enable down-sampling method.

1. Adjust parameters in `./exp/DeepAFP/trainer.py`.

2. Run experiment:
    ```bash
    python ./exp/DeepAFP/trainer.py
    ```