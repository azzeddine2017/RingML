# File: examples/classify_demo.ring
# Description: Multi-class classification example

load "../src/ringml.ring"
decimals(8) 
see "=== RingML Multi-Class Classification Demo ===" + nl

# 1. Data: 3 Samples, 4 Features each
# Sample 1: [1,1,0,0] -> Class 1 [1,0,0]
# Sample 2: [0,0,1,1] -> Class 2 [0,1,0]
# Sample 3: [0,1,1,0] -> Class 3 [0,0,1]

inputs = new Tensor(3, 4)
inputs.aData = [
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 0]
]

targets = new Tensor(3, 3) # One-Hot Encoded
targets.aData = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

# 2. Model
model = new Sequential
model.add(new Dense(4, 8)) # Input 4 -> Hidden 8 (Increased neurons)
model.add(new ReLU)
model.add(new Dense(8, 3)) # Hidden 8 -> Output 3
model.add(new Softmax)     

# 3. Training
criterion = new CrossEntropyLoss
optimizer = new SGD(0.1)

see "Training..." + nl
for epoch = 1 to 5000
    # Forward
    preds = model.forward(inputs)
    
    # Loss
    loss = criterion.forward(preds, targets)
    
    if epoch % 1000 = 0
        see "Epoch " + epoch + " Loss: " + loss + nl
    ok
    
    # Backward
    grad = criterion.backward(preds, targets)
    model.backward(grad)
    
    # Update
    for layer in model.getLayers()
        optimizer.update(layer)
    next
next

see "=== Final Predictions ===" + nl
final = model.forward(inputs)
final.print()

see "=== Expected Targets ===" + nl
targets.print()