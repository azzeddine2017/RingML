# RingML Examples

This directory contains various examples demonstrating how to use the RingML library for different tasks, from simple binary classification to more complex scenarios.

## 📂 Directory Structure

### 1. Basic Examples
These scripts demonstrate core functionalities of the library.

- **`xor_train.ring`**: The "Hello World" of Deep Learning. Trains a simple network to solve the XOR problem using binary classification.
- **`classify_demo.ring`**: Demonstrates multi-class classification using `Softmax` and `CrossEntropyLoss`.
- **`classify_demo2.ring`**: A variation of the classification demo.
- **`save_load_demo.ring`**: Shows how to save a trained model's weights to a file (`.rdata`) and load them back for inference.
- **`loader_demo.ring`**: Demonstrates how to use the `DataLoader` class for efficient mini-batch processing.

### 2. Advanced Applications

#### ♟️ Chess End-Game (`examples/Chess_End_Game/`)
A real-world application that predicts the result of a Chess End-Game (King + Rook vs. King).
- **Dataset**: Uses the KRK dataset (`chess.csv`).
- **Scripts**:
  - `chess_train_split.ring`: Training script with Train/Validation data split to monitor overfitting.
  - `chess_train_fast.ring`: Optimized training script using `DataLoader`.
  - `chess_train_adam.ring`: Training using the `Adam` optimizer.
  - `chess_app.ring`: Interactive application to predict game results from board coordinates.
  - `chess_dataset.ring`: Custom `Dataset` implementation for parsing chess positions.
  - `chess_utils.ring`: Utility functions for the chess example.

#### 🔢 MNIST (`examples/mnist/`)
(In Progress) Implementation for handwritten digit recognition using the MNIST dataset.

## 🚀 How to Run
To run any example, navigate to the project root directory and execute the script using the Ring interpreter.

**Example: Running XOR Training**
```bash
ring examples/xor_train.ring
```

**Example: Running Chess Training**
```bash
ring examples/Chess_End_Game/chess_train_fast.ring
```
