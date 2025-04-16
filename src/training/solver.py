"""A Solver class implementing training and inference logic for time series classification tasks.
"""
from tqdm import tqdm
from typing import List, Optional, Tuple

from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Solver:
    """
    Hidden states are computed via a single forward pass on the randomized ResRMN. Then, a simple 
    linear readout (RidgeClassifier) is trained on the hidden states and used for inference.

    Attributes:
        classifier: Linear readout layer (RidgeClassifier) used for classification.
        train_accuracy: Accuracy achieved on the training set.
        test_accuracy: Accuracy achieved on the test set.
    """
    def __init__(
        self, 
        device: torch.device,
        model: nn.Module, 
        train_dataloader: DataLoader = None, 
        test_dataloader: DataLoader = None, 
        permutation: Optional[List[int]] = None,
        reg: float = 0,
    ):
        """
        Args:
            permutation: Permutation used to shuffle MNIST pixels for psMNIST.
            reg: Readout layer regularization strength.
        """
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        # Check dataset features and sequence length
        dummy = next(iter(train_dataloader))[0]
        if len(dummy.shape) == 4: # psMNIST
            self.seqlen = 784
            self.features_dim = 1
        else:
            self.seqlen = dummy.shape[-1]
        self.features_dim = dummy.shape[1]
            
        self.permutation = None
        if permutation:
            self.permutation = torch.Tensor(permutation).to(torch.long).to(device)

        self.classifier = RidgeClassifier(alpha=reg, solver='svd')
        
        self._reset()
    
    def _reset(self) -> None:
        """Reset solver state."""
        self.train_accuracy = None
        self.test_accuracy = None

    @torch.no_grad()
    def evaluate(
        self, 
        dataloader: DataLoader, 
        scaler, 
    ) -> Tuple[float, float]:
        self.model.eval()

        # Foward pass to collect hidden states
        states, targets = [], []
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(self.device)
                x = x.reshape(x.shape[0], self.features_dim, self.seqlen) 
                x = x.permute(0, 2, 1)
                if self.permutation is not None: # only for psMNIST
                    x = x[:, self.permutation, :]
                _, out = self.model(x)
                states.append(out.cpu())
                targets.append(y)
        states = torch.cat(states, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        # Evaluate
        states = scaler.transform(states)
        return self.classifier.score(states, targets)
    
    @torch.no_grad()
    def train(self) -> Tuple[float, float]:
        self.model.to(self.device)

        # Foward pass to collect hidden states
        states, targets = [], []
        with torch.no_grad():
            for x, y in tqdm(self.train_dataloader):
                x = x.to(self.device)
                # MNIST image to sequence format
                x = x.reshape(x.shape[0], self.features_dim, self.seqlen)
                x = x.permute(0, 2, 1)
                if self.permutation is not None:
                    x = x[:, self.permutation, :]
                _, out = self.model(x)
                states.append(out.cpu())
                targets.append(y)
        states = torch.cat(states, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        # Train the readout
        scaler = preprocessing.StandardScaler().fit(states)
        states = scaler.transform(states)
        self.classifier.fit(states, targets)

        # Evaluate
        self.train_accuracy = self.classifier.score(states, targets)
        self.test_accuracy = self.evaluate(self.test_dataloader, scaler)

        print(
            f"Train accuracy: {self.train_accuracy:.4f} - Test accuracy: {self.test_accuracy:.4f}"
        )

        return (self.train_accuracy, self.test_accuracy)



        