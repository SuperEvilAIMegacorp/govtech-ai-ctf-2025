#!/usr/bin/env python3
"""
model.py

Contains the shared model architecture used by both train_cpu.py and verifier.py.
"""

import torch.nn as nn


class SmallFashionCNN(nn.Module):
    """
    Small CNN for Fashion-MNIST classification.
    
    Architecture:
    - Conv2d(1, 32, 3, padding=1) + ReLU + MaxPool2d(2)  # 28 -> 14
    - Conv2d(32, 64, 3, padding=1) + ReLU + MaxPool2d(2)  # 14 -> 7
    - Flatten
    - Linear(64 * 7 * 7, 128) + ReLU
    - Linear(128, num_classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
