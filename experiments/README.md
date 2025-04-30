# Experiments
Here, you can find instructions for reproducing analyses and experiments presented in the manuscripts.

# Eigenspectrum analysis
In `eigenspectrum.ipynb` you can find a small tutorial to replicate the eigenspectrum analyses presented in Figure 2, for the orthogonal matrices, and in Figure 3, for ResRMN's Jacobian.

## Classification
In `residualrmn_configs.sh`, for each ResRMN configuration and dataset, you can find the optimal configurations identified during model selection and used in our experiments.
Before running any test, make sure to download the relevant files from [https://www.timeseriesclassification.com/](https://www.timeseriesclassification.com/) and unzip them in `datasets/<dataset_name>`.

For instance, to test ResRMN on the **Adiac** dataset across $10$ different random initializations, run the following:

### ResRMN-R
```
python test_residualrmn.py --dataset adiac --skip_option_t ortho --n_units_m 100 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 1.0 --bias_scaling 0.0 --rho 1.1 --alpha 1.0 --beta 0.5 --reg 10.0 --batch_size 1024 --n_trials 10
```

### ResRMN-C
```
python test_residualrmn.py --dataset adiac --skip_option_t cycle --n_units_m 100 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 1.0 --bias_scaling 1.0 --rho 1.0 --alpha 1.0 --beta 0.5 --reg 10.0 --batch_size 1024 --n_trials 10
```

### ResRMN-I
```
python test_residualrmn.py --dataset adiac --skip_option_t identity --n_units_m 100 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 1.0 --bias_scaling 0.01 --rho 1.1 --alpha 1.0 --beta 0.01 --reg 0.1 --batch_size 1024 --n_trials 10
```