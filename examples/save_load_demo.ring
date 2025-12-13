load "../src/ringml.ring"
decimals(4)

see "=== RingML Save/Load Model Demo ===" + nl

# 1. Prepare Data (XOR)
inputs  = new Tensor(4, 2) { aData = [[0,0], [0,1], [1,0], [1,1]] }
targets = new Tensor(4, 1) { aData = [[0],   [1],   [1],   [0]] }

# 2. Train Model A
see "--> Training Model A..." + nl
modelA = new Sequential
modelA.add(new Dense(2, 4))
modelA.add(new Sigmoid)
modelA.add(new Dense(4, 1))
modelA.add(new Sigmoid)

optimizer = new SGD(0.5)
criterion = new MSELoss

for epoch = 1 to 20000
    preds = modelA.forward(inputs)
    grad  = criterion.backward(preds, targets)
    modelA.backward(grad)
    for l in modelA.getLayers() optimizer.update(l) next
    if epoch % 1000 = 0 see "." ok
next

see "Model A Predictions (Trained):" + nl
predA = modelA.forward(inputs)
predA.print()

# 3. Save Model A
cFile = "xor_weights.rdata"
modelA.saveWeights(cFile)

# 4. Create Model B (Untrained)
see nl + "--> Creating Model B (Same Architecture, Untrained)..." + nl
modelB = new Sequential
modelB.add(new Dense(2, 4))
modelB.add(new Sigmoid)
modelB.add(new Dense(4, 1))
modelB.add(new Sigmoid)

see "Model B Predictions (Before Loading):" + nl
predB = modelB.forward(inputs)
predB.print()

# 5. Load Weights into Model B
modelB.loadWeights(cFile)

see "Model B Predictions (After Loading):" + nl
predB = modelB.forward(inputs)
predB.print()

# 6. Verification
see "--> Verification: "
# Simple check of first element
diff = fabs(predA.aData[1][1] - predB.aData[1][1])
if diff < 0.0001
    see "SUCCESS (Models Match)" + nl
else
    see "FAILURE (Mismatch)" + nl
ok