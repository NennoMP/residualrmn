#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch

import src.data.data_utils as data_utils
import src.networks.residualrmn as residualrmn
import src.training.solver as solver

DATA_DIR = '../datasets' # directory where datasets should be stored


parser = argparse.ArgumentParser(description='hparams')
# Training setup
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch_size', type=int, default=1024, help='mini-batch size (default: 1024)')
parser.add_argument('--n_trials', type=int, default=10, help='number of trials (default: 10)')
# Hyperparameters
parser.add_argument('--skip_option_t', type=str, help='temporal residual connections configuration')
parser.add_argument('--n_units_m', type=int, default=100, help='memory reservoir units (default: 100)')
parser.add_argument('--n_units', type=int, default=100, help='non-linear reservoir units (default: 100)')
parser.add_argument('--in_scaling_m', type=float, default=1., help='memory reservoir input kernel scaling (default: 1.0)')
parser.add_argument('--bias_scaling_m', type=float, default=0., help='memory reservoir bias vector scaling (default: 0.0)')
parser.add_argument('--in_scaling', type=float, default=1., help='non-linear reservoir input kernel scaling (default: 1.0)')
parser.add_argument('--memory_scaling', type=float, default=1., help='non-linear reservoir memory kernel scaling (default: 1.0)')
parser.add_argument('--bias_scaling', type=float, default=0., help='non-linear reservoir bias vector scaling (default: 0.0)')
parser.add_argument('--rho', type=float, default=1., help='spectral radius (default: 1.0)')
parser.add_argument('--alpha', type=float, default=1., help='residual branch coefficient (default: 1.0)')
parser.add_argument('--beta', type=float, default=1., help='non-linear branch coefficient (default: 1.0)')
parser.add_argument('--reg', type=float, default=0., help='readout regularization strength (default: 0.0)')


def main() -> None:
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    permutation = None
    if args.dataset == 'psMNIST':
        full_train_dataloader, _, _, test_dataloader = data_utils.get_mnist(
            data_dir=f'{DATA_DIR}',
            batch_size=args.batch_size
        )
        # Load permutation
        with open(f'{DATA_DIR}/MNIST/permutation.json') as inf:
            permutation = json.load(inf)
    else:
        full_train_dataloader, _, _, test_dataloader = data_utils.get_classification_task(
            data_dir=f'{DATA_DIR}/{args.dataset}',
            dataset_name=args.dataset,
            batch_size=args.batch_size
        )
    in_size = next(iter(full_train_dataloader))[0].shape[1]

    test_accuracies = []
    for i in range(args.n_trials):
        print(f'----- TRIAL {i+1}/{args.n_trials}')

        # Initialize ResRMN
        hparams = vars(args)
        hparams['in_size'] = in_size
        model = residualrmn.residualrmn_(
            hparams=hparams,
        )

        # Freeze all weights of ResRMN
        # Note: @torch.no_grad() decorator is used in the Solver training logicthus
        # thus, this is not strictly necessary
        for param in model.parameters():
            param.requires_grad = False

        # Training
        solver_ = solver.Solver(
            device=device,
            model=model,
            train_dataloader=full_train_dataloader,
            test_dataloader=test_dataloader,
            permutation=permutation,
            reg=args.reg
        )
        _, test_accuracy = solver_.train()
        test_accuracies.append(test_accuracy)

    test_mean, test_std = np.mean(test_accuracies), np.std(test_accuracies)
    n_params = solver_.classifier.coef_.size + solver_.classifier.intercept_.size
    print(
        '##################################################\n'
        f'Mean test accuracy: {test_mean:.4f} ± {test_std:.4f}\n'
        f'Number of trainable parameters: {n_params}\n'
        '##################################################'
    )

    save_experiment(
        model_name=residualrmn.ResidualRMN.__name__.lower(),
        n_params=n_params,
        test_mean=test_mean, 
        test_std=test_std, 
        args=args,
    )

def save_experiment(
    model_name: str,
    n_params: int,
    test_mean: float, 
    test_std: float, 
    args: argparse.Namespace
) -> None:
    """
    Args:
        model_name: name of the model class
        n_params: number of trainable parameters
        means: tuple of averages on metrics of interest
        stds: tuple of standard deviations on metrics of interest
    """
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'n_params': n_params,
        'config': vars(args),
        'results': {
            'test_accuracy': {'mean': f'{test_mean:.4f}', 'std': f'{test_std:.4f}'},
        }
    }

    os.makedirs('results', exist_ok=True)
    out_file = f'results/{model_name}_{args.dataset}.csv'
    with open(out_file, 'a', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=output.keys())
        writer.writerow(output)
    print(f'Saved results to {out_file}.')


if __name__ == '__main__':
    main()