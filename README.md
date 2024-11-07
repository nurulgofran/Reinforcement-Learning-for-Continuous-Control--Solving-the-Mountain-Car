# Reinforcement Learning for Mountain Car

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

## Overview
Implementation of Q-learning algorithm to solve OpenAI Gym's Mountain Car environment, where an underpowered car must learn to reach the mountaintop through momentum.

## Features
- Q-learning with discretized state space
- Real-time training visualization
- Performance metrics and plots
- Customizable hyperparameters

## Technical Details
- **Environment**: Mountain Car (OpenAI Gym)
- **State Space**: Position [-1.2, 0.6] and Velocity [-0.07, 0.07]
- **Action Space**: [Push left, No push, Push right]
- **Hyperparameters**:
  - Learning Rate: 0.1
  - Discount Factor: 0.95
  - Episodes: 2000

## Installation

```bash
# Clone repository
git clone https://github.com/nurulgofran/Reinforcement-Learning-for-Continuous-Control--Solving-the-Mountain-Car.git
cd Reinforcement-Learning-for-Continuous-Control--Solving-the-Mountain-Car

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


## Usage
```bash
# Run training
python mountain_car.py

.
├── [mountain_car.py](http://_vscodecontentref_/1)     # Main implementation
├── [requirements.txt](http://_vscodecontentref_/2)    # Dependencies
├── LICENSE            # MIT license
└── [README.md](http://_vscodecontentref_/3)         # Documentation



Implementation Details
State space discretization for Q-learning
Epsilon-greedy exploration strategy
Real-time reward tracking
Learning progress visualization
Contributing
Fork repository
Create feature branch (git checkout -b feature/NewFeature)
Commit changes (git commit -m 'Add NewFeature')
Push to branch (git push origin feature/NewFeature)
Submit Pull Request