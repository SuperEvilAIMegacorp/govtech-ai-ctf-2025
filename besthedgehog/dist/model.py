#!/usr/bin/env python3
"""
PyTorch model for evaluating Best Hedgehog performance.

This model takes the 6 category scores as input and outputs a final evaluation score.
The goal is to manipulate the training data so that 'jaga' gets a score of 100.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HedgehogEvaluationModel(nn.Module):
    """Neural network to evaluate hedgehog performance based on qualities."""
    
    def __init__(self, input_size=6, hidden_size=32):
        super(HedgehogEvaluationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(training_data, target_scores=None, epochs=100):
    """
    Train the hedgehog evaluation model.
    
    Args:
    training_data: List of 6 qualities [furriness, cuteness, friendliness, curiosity, agility, sleepiness]
        target_scores: Optional list of target evaluation scores for supervised learning
                      If None, synthetic targets will be created
        epochs: Number of training epochs
    
    Returns:
        Trained model
    """
    # Convert training data to tensors
    X = torch.FloatTensor(training_data)
    
    if target_scores is not None:
        # Use provided target scores for supervised learning
        targets = [[score] for score in target_scores]
        y = torch.FloatTensor(targets)
    else:
    # Create synthetic targets based on average qualities (legacy mode)
        # This creates a correlation between input scores and output evaluation
        targets = []
        for scores in training_data:
            # Target is weighted average of scores with some variation
            avg_score = np.mean(scores)
            # Add some randomness but keep it predictable
            target = avg_score + np.random.normal(0, 5)
            target = max(0, min(100, target))  # Clamp to 0-100 range
            targets.append([target])
        
        y = torch.FloatTensor(targets)
    
    # Initialize model
    model = HedgehogEvaluationModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model

def evaluate_hedgehog(model, hedgehog_qualities):
    """
    Evaluate a single hedgehog using the trained model.

    Args:
    model: Trained HedgehogEvaluationModel
        hedgehog_qualities: List of 6 qualities [furriness, cuteness, friendliness, curiosity, agility, sleepiness]

    Returns:
        Evaluation score (0-100)
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor([hedgehog_qualities])
        output = model(input_tensor)
        return output.item()

def create_optimal_scores_for_target(target_score=100):
    """
    Helper function to determine what scores would yield the target evaluation.
    This is useful for understanding model behavior and score relationships.
    
    Args:
        target_score: Desired evaluation score
    
    Returns:
        Suggested scores that should yield the target evaluation
    """
    # Since the model uses weighted averages, high scores across all categories
    # should yield a high evaluation score
    if target_score >= 100:
        return [100, 100, 100, 100, 100, 100]
    elif target_score >= 90:
        return [95, 95, 95, 95, 95, 95]
    else:
        base_score = int(target_score * 0.9)
        return [base_score] * 6

if __name__ == '__main__':
    # Test the model with sample qualities
    sample_data = [
        [85, 92, 78, 65, 88, 79],
        [76, 84, 89, 91, 73, 82],
        [94, 67, 85, 78, 92, 88],
    [45, 50, 40, 35, 48, 42]  # jaga's initial scores
    ]
    
    model = train_model(sample_data)
    
    # Test evaluation
    jaga_scores = [45, 50, 40, 35, 48, 42]
    evaluation = evaluate_hedgehog(model, jaga_scores)
    print(f"Jaga's current evaluation: {evaluation:.2f}")
    
    # Show what scores would get 100
    optimal_scores = create_optimal_scores_for_target(100)
    optimal_evaluation = evaluate_hedgehog(model, optimal_scores)
    print(f"With optimal scores {optimal_scores}: {optimal_evaluation:.2f}")