"""Module containing utils for data loading and pre-processing."""
import os
from typing import Tuple

import numpy as np
import torch
import torchvision
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


def get_classification_task(
    dataset_name: str,
    data_dir: str = 'datasets', 
    batch_size: int = 128, 
    val_split: float = 0.3,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Load a time series classification dataset from https://www.timeseriesclassification.com/.
    
    Args:
        data_dir: Directory where files should be downloaded/extracted
        dataset_name: Name of the dataset
        val_split: Percentage of training data to use for validation
        random_state: Random state for train-val splits reproducibility

    Returns:
        A tuple of dataloaders for (entire) training, train-val splits, and test data.
    
    Raises:
        ValueError: If train or test data is not found in the specified directory.
    """
    train_dataset, train_target = load_classification(
        dataset_name, 
        split='train', 
        extract_path=data_dir
    )
    train_dataset = np.asarray(train_dataset)
    train_target = np.asarray(train_target)
    print(f'[{dataset_name}] Training data shape (N, X, T): {train_dataset.shape}')

    test_dataset, test_target = load_classification(
        dataset_name, 
        split='test', 
        extract_path=data_dir
    )
    test_dataset = np.asarray(test_dataset)
    test_target = np.asarray(test_target)
    print(f'[{dataset_name}] Test data shape (N, X, T): {test_dataset.shape}')

    # Train-Val stratified split
    full_train_dataset = train_dataset
    full_train_target = train_target
    train_dataset, val_dataset, train_target, val_target = train_test_split(
        full_train_dataset,
        full_train_target,
        test_size=val_split,
        stratify=full_train_target,
        random_state=random_state,
    )
    
    # Convert string labels to numerical indexes
    label_encoder = LabelEncoder()
    full_train_target = label_encoder.fit_transform(full_train_target)
    train_target = label_encoder.transform(train_target)
    val_target = label_encoder.transform(val_target)
    test_target = label_encoder.transform(test_target)
    
    # Convert to tensors
    full_train_dataset = torch.FloatTensor(full_train_dataset)
    full_train_target = torch.LongTensor(full_train_target)
    train_dataset = torch.FloatTensor(train_dataset)
    train_target = torch.LongTensor(train_target)

    if not isinstance(val_dataset, torch.Tensor):
        val_dataset = torch.FloatTensor(val_dataset)
        test_dataset = torch.FloatTensor(test_dataset)
    if not isinstance(val_target, torch.Tensor):
        val_target = torch.LongTensor(val_target)
        test_target = torch.LongTensor(test_target)

    # Dataloaders
    full_train_dataset = TensorDataset(full_train_dataset, full_train_target)
    train_dataset = TensorDataset(train_dataset, train_target)
    val_dataset = TensorDataset(val_dataset, val_target)
    test_dataset = TensorDataset(test_dataset, test_target)   
    full_train_dataloader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    return (full_train_dataloader, train_dataloader, val_dataloader, test_dataloader)

def get_mnist(
    data_dir: str, 
    batch_size: int = 128, 
    val_split: float = 0.05,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Load the MNIST dataset.
    
    Args:
        data_dir: Directory for downloading/storing MNIST
        val_split: Percentage of training data to use for validation

    Returns:
        A tuple of dataloaders for (entire) training, train-val splits, and test data.
    """
    full_train_dataset = torchvision.datasets.MNIST(
        data_dir, 
        train=True, 
        transform=torchvision.transforms.ToTensor(), 
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        data_dir, 
        train=False, 
        transform=torchvision.transforms.ToTensor(), 
        download=True
    )
    
    # Train-Val split
    val_size = np.int64(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
    )
    
    # Dataloaders
    full_train_dataloader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    return (full_train_dataloader, train_dataloader, val_dataloader, test_dataloader)