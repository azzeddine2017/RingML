load "../src/ringml.ring"

see "=== Dimensions Diagnosis ===" + nl

# 1. Create Data
inputs  = new Tensor(1, 2) { aData = [[0.5, 0.1]] } # (1, 2)
targets = new Tensor(1, 1) { aData = [[0.7]] }      # (1, 1)

# 2. Create Model Layer 2 (The problematic one)
# Layer 1: (2 inputs -> 4 outputs)
l1 = new Dense(2, 4) 
# Layer 2: (4 inputs -> 1 output) -> Weights should be (4, 1)
l2 = new Dense(4, 1)

see "Initial Weights L2 Shape: (" + l2.oWeights.nRows + ", " + l2.oWeights.nCols + ")" + nl
see "Row 1 Len: " + len(l2.oWeights.aData[1]) + nl

# 3. Simulate Forward
out1 = l1.forward(inputs) # Output is (1, 4)
out2 = l2.forward(out1)   # Output is (1, 1)

# 4. Simulate Backward
crit = new MSELoss
gradLoss = crit.backward(out2, targets) # (1, 1)

see "--- Backward Step ---" + nl
gradIn2 = l2.backward(gradLoss)

# Check Gradients Shape
gW = l2.oGradWeights
see "Gradients L2 Shape: (" + gW.nRows + ", " + gW.nCols + ")" + nl
see "Row 1 Len: " + len(gW.aData[1]) + nl

if gW.nRows = 1 and gW.nCols = 4
    see "CRITICAL: Gradients are Transposed (1, 4) instead of (4, 1)!" + nl
    see "This is causing the corruption." + nl
ok

# 5. Simulate Update
opt = new SGD(0.1)
opt.update(l2)

see "--- After Update ---" + nl
see "Weights L2 Shape: (" + l2.oWeights.nRows + ", " + l2.oWeights.nCols + ")" + nl
see "Row 1 Len: " + len(l2.oWeights.aData[1]) + nl
see "Data: " 
l2.oWeights.print()