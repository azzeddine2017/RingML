# File: examples/xor_train.ring
# Description: Training a Neural Network to solve XOR problem
# Author: Code Gear-1

# Load RingML Library (Adjust path as needed)
load "../src/ringml.ring"

decimals(10)

see "=== RingML XOR Training Example ===" + nl

# 1. Prepare Data (4 samples, 2 features)
# XOR Inputs: [0,0], [0,1], [1,0], [1,1]
inputs = new Tensor(4, 2)
inputs.aData = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

# XOR Targets: [0], [1], [1], [0]
targets = new Tensor(4, 1)
targets.aData = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]

# 2. Build Model
# Architecture: Input(2) -> Hidden(4) -> Output(1)
model = new Sequential

# Hidden Layer: 2 inputs -> 4 neurons, followed by Sigmoid
model.add(new Dense(2, 4))
model.add(new Sigmoid)

# Output Layer: 4 inputs -> 1 neuron, followed by Sigmoid
model.add(new Dense(4, 1))
model.add(new Sigmoid)

# 3. Setup Loss and Optimizer
criterion = new MSELoss
# Use higher learning rate for XOR with Sigmoid to speed up convergence
optimizer = new SGD(0.5) 

# 4. Training Loop
nEpochs = 5000

see "Training started for " + nEpochs + " epochs..." + nl

for epoch = 1 to nEpochs
    # A. Forward Pass
    preds = model.forward(inputs)
    
    # B. Calculate Loss
    loss = criterion.forward(preds, targets)
    
    # Print progress every 500 epochs
    if epoch % 500 = 0
        see "Epoch " + epoch + " : Loss = " + loss + nl
    ok
    
    # C. Backward Pass
    lossGrad = criterion.backward(preds, targets)
    model.backward(lossGrad)
    
    # D. Optimize (Update Weights)
    for layer in model.getLayers()
        optimizer.update(layer)
    next
next

see nl + "=== Training Finished ===" + nl

# 5. Verify Predictions
see "Final Predictions:" + nl
finalPreds = model.forward(inputs)
finalPreds.print()

see "Expected Targets:" + nl
targets.print()