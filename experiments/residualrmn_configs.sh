# Best configurations for each ResRMN configuration

#==============================
# Adiac
#==============================
# ResRMN-R
python test_residualrmn.py --dataset Adiac --skip_option_t ortho --n_units_m 176 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 1.0 --bias_scaling 0.0 --rho 1.1 --alpha 1.0 --beta 0.5 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset Adiac --skip_option_t cycle --n_units_m 176 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 1.0 --bias_scaling 1.0 --rho 1.0 --alpha 1.0 --beta 0.5 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset Adiac --skip_option_t identity --n_units_m 176 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 1.0 --bias_scaling 0.01 --rho 1.1 --alpha 1.0 --beta 0.01 --reg 0.1 --batch_size 1024 --n_trials 10

#==============================
# Beef
#==============================
# ResRMN-R
python test_residualrmn.py --dataset Beef --skip_option_t ortho --n_units_m 470 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 0.01 --bias_scaling 0.0 --rho 1.0 --alpha 0.0 --beta 0.9 --reg 0.1 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset Beef --skip_option_t cycle --n_units_m 470 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.1 --bias_scaling 0.01 --rho 0.9 --alpha 0.01 --beta 0.1 --reg 1.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset Beef --skip_option_t identity --n_units_m 470 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 0.01 --bias_scaling 0.0 --rho 0.9 --alpha 0.99 --beta 0.01 --reg 0.1 --batch_size 1024 --n_trials 10

#==============================
# Blink
#==============================
# ResRMN-R
python test_residualrmn.py --dataset Blink --skip_option_t ortho --n_units_m 510 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 0.01 --bias_scaling 0.0 --rho 1.1 --alpha 0.9 --beta 0.1 --reg 100.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset Blink --skip_option_t cycle --n_units_m 510 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.1 --bias_scaling 1.0 --rho 1.1 --alpha 1.0 --beta 1.0 --reg 100.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset Blink --skip_option_t identity --n_units_m 510 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.1 --bias_scaling 0.1 --rho 1.1 --alpha 1.0 --beta 0.01 --reg 10.0 --batch_size 1024 --n_trials 10

#==============================
# Car
#==============================
# ResRMN-R
python test_residualrmn.py --dataset Car --skip_option_t ortho --n_units_m 577 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.01 --bias_scaling 0.01 --rho 1.1 --alpha 0.5 --beta 0.5 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset Car --skip_option_t cycle --n_units_m 577 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.1 --bias_scaling 0.0 --rho 1.0 --alpha 0.1 --beta 0.9 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset Car --skip_option_t identity --n_units_m 577 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 1.0 --bias_scaling 0.01 --rho 1.0 --alpha 1.0 --beta 0.1 --reg 10.0 --batch_size 1024 --n_trials 10

#==============================
# DuckDuckGeese
#==============================
# ResRMN-R
python test_residualrmn.py --dataset DuckDuckGeese --skip_option_t ortho --n_units_m 270 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.01 --bias_scaling 0.1 --rho 0.9 --alpha 0.1 --beta 0.5 --reg 100.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset DuckDuckGeese --skip_option_t cycle --n_units_m 270 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.01 --bias_scaling 0.0 --rho 0.9 --alpha 0.1 --beta 0.9 --reg 100.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset DuckDuckGeese --skip_option_t identity --n_units_m 270 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.01 --bias_scaling 0.1 --rho 1.0 --alpha 1.0 --beta 0.01 --reg 100.0 --batch_size 1024 --n_trials 10

#==============================
# FordA
#==============================
# ResRMN-R
python test_residualrmn.py --dataset FordA --skip_option_t ortho --n_units_m 500 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.01 --bias_scaling 1.0 --rho 1.1 --alpha 1.0 --beta 0.1 --reg 1.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset FordA --skip_option_t cycle --n_units_m 500 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.1 --bias_scaling 1.0 --rho 0.9 --alpha 0.01 --beta 1.0 --reg 0.1 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset FordA --skip_option_t identity --n_units_m 500 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.01 --bias_scaling 1.0 --rho 1.1 --alpha 1.0 --beta 0.01 --reg 0.1 --batch_size 1024 --n_trials 10

#==============================
# FordB
#==============================
# ResRMN-R
python test_residualrmn.py --dataset FordB --skip_option_t ortho --n_units_m 500 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 0.01 --bias_scaling 0.1 --rho 1.0 --alpha 1.0 --beta 0.1 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset FordB --skip_option_t cycle --n_units_m 500 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.01 --bias_scaling 1.0 --rho 1.1 --alpha 1.0 --beta 0.1 --reg 0.1 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset FordB --skip_option_t identity --n_units_m 500 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 1.0 --bias_scaling 1.0 --rho 1.1 --alpha 1.0 --beta 0.01 --reg 100.0 --batch_size 1024 --n_trials 10

#==============================
# HandMovementDirection
#==============================
# ResRMN-R
python test_residualrmn.py --dataset HandMovementDirection --skip_option_t ortho --n_units_m 400 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 0.01 --bias_scaling 0.01 --rho 1.1 --alpha 0.5 --beta 0.1 --reg 100.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset HandMovementDirection --skip_option_t cycle --n_units_m 400 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.01 --bias_scaling 0.1 --rho 1.0 --alpha 0.5 --beta 0.01 --reg 100.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset HandMovementDirection --skip_option_t identity --n_units_m 400 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.1 --bias_scaling 0.0 --rho 1.1 --alpha 0.9 --beta 0.99 --reg 100.0 --batch_size 1024 --n_trials 10

#==============================
# Libras
#==============================
# ResRMN-R
python test_residualrmn.py --dataset Libras --skip_option_t ortho --n_units_m 45 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 1.0 --bias_scaling 0.0 --rho 0.9 --alpha 0.99 --beta 0.5 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset Libras --skip_option_t cycle --n_units_m 45 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 1.0 --bias_scaling 1.0 --rho 1.0 --alpha 0.9 --beta 0.1 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset Libras --skip_option_t identity --n_units_m 45 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.1 --bias_scaling 0.0 --rho 0.9 --alpha 0.99 --beta 0.01 --reg 0.01 --batch_size 1024 --n_trials 10

#==============================
# Mallat
#==============================
# ResRMN-R
python test_residualrmn.py --dataset Mallat --skip_option_t ortho --n_units_m 1024 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 1.0 --bias_scaling 0.0 --rho 1.1 --alpha 0.1 --beta 0.01 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset Mallat --skip_option_t cycle --n_units_m 1024 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 1.0 --bias_scaling 1.0 --rho 0.9 --alpha 0.5 --beta 1.0 --reg 10.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset Mallat --skip_option_t identity --n_units_m 1024 --n_units 100 --in_scaling_m 1.0 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.1 --bias_scaling 0.01 --rho 0.9 --alpha 0.99 --beta 0.1 --reg 10.0 --batch_size 1024 --n_trials 10

#==============================
# OSULeaf
#==============================
# ResRMN-R
python test_residualrmn.py --dataset OSULeaf --skip_option_t ortho --n_units_m 427 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.01 --bias_scaling 1.0 --rho 1.0 --alpha 0.5 --beta 0.1 --reg 1.0 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset OSULeaf --skip_option_t cycle --n_units_m 427 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.01 --bias_scaling 0.1 --rho 0.9 --alpha 0.5 --beta 0.5 --reg 0.01 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset OSULeaf --skip_option_t identity --n_units_m 427 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.1 --bias_scaling 1.0 --rho 1.0 --alpha 1.0 --beta 0.01 --reg 10.0 --batch_size 1024 --n_trials 10

#==============================
# Wine
#==============================
# ResRMN-R
python test_residualrmn.py --dataset Wine --skip_option_t ortho --n_units_m 234 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 0.1 --bias_scaling 0.01 --rho 0.9 --alpha 0.9 --beta 0.01 --reg 0.01 --batch_size 1024 --n_trials 10
# ResRMN-C
python test_residualrmn.py --dataset Wine --skip_option_t cycle --n_units_m 234 --n_units 100 --in_scaling_m 0.1 --bias_scaling_m 0.0 --in_scaling 0.1 --memory_scaling 0.01 --bias_scaling 1.0 --rho 0.9 --alpha 0.5 --beta 0.9 --reg 0.0 --batch_size 1024 --n_trials 10
# ResRMN-I
python test_residualrmn.py --dataset Wine --skip_option_t identity --n_units_m 234 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 0.01 --memory_scaling 0.1 --bias_scaling 0.01 --rho 0.9 --alpha 0.9 --beta 0.01 --reg 0.0 --batch_size 1024 --n_trials 10

#==============================
# psMNIST
#==============================
# ResRMN-R
python test_residualrmn.py --dataset psMNIST --skip_option_t ortho --n_units_m 784 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 1.0 --bias_scaling 0.01 --rho 1.0 --alpha 0.5 --beta 0.01 --reg 0.01 --batch_size 1024 --n_trials 5
# ResRMN-C
python test_residualrmn.py --dataset psMNIST --skip_option_t cycle --n_units_m 784 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.1 --bias_scaling 0.1 --rho 0.9 --alpha 0.1 --beta 1.0 --reg 0.01 --batch_size 1024 --n_trials 5
# ResRMN-I
python test_residualrmn.py --dataset psMNIST --skip_option_t identity --n_units_m 784 --n_units 100 --in_scaling_m 0.01 --bias_scaling_m 0.0 --in_scaling 1.0 --memory_scaling 0.1 --bias_scaling 0.1 --rho 1.0 --alpha 0.99 --beta 0.01 --reg 0.01 --batch_size 1024 --n_trials 5
