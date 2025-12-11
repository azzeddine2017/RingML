# File: examples/classify_demo.ring
# Description: Multi-class classification example (Improved)

load "../src/ringml.ring"
decimals(8) 

see "=== RingML Multi-Class Classification Demo ===" + nl

# 1. Data
inputs = new Tensor(3, 4)
inputs.aData = [
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 0]
]

targets = new Tensor(3, 3) 
targets.aData = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

# 2. Model
model = new Sequential
model.add(new Dense(4, 8)) 
# FIX: Use Sigmoid instead of ReLU for small datasets to avoid "Dying ReLU"
model.add(new Sigmoid)     
model.add(new Dense(8, 3)) 
model.add(new Softmax)     

# 3. Training
criterion = new CrossEntropyLoss
optimizer = new SGD(0.5) # Increased learning rate for Sigmoid

see "Training..." + nl
for epoch = 1 to 3000
    # Forward
    preds = model.forward(inputs)
    
    # Loss
    loss = criterion.forward(preds, targets)
    
    if epoch % 500 = 0
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