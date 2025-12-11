load "../src/ringml.ring"
decimals(8) 
see "=== DataLoader Demo ===" + nl

# 1. Create Dummy Data (10 Samples)
inputs = new Tensor(10, 2)
targets = new Tensor(10, 1)

# Fill with dummy data
for i = 1 to 10
    inputs.aData[i] = [i, i*2]
    targets.aData[i] = [1]
next

# 2. Wrap in Dataset
dataset = new TensorDataset(inputs, targets)

# 3. Create DataLoader (Batch Size = 4)
# This should create 3 batches: (4 samples, 4 samples, 2 samples)
loader = new DataLoader(dataset, 4)

see "Total Samples: " + dataset.len() + nl
see "Batch Size:    " + loader.nBatchSize + nl
see "Num Batches:   " + loader.nBatches + nl + nl

# 4. Iterate Batches
for b = 1 to loader.nBatches
    see "--- Batch " + b + " ---" + nl
    batchData = loader.getBatch(b)
    batchX = batchData[1]
    batchY = batchData[2]
    
    see "Input Shape: " batchX.print()
    # Here you would typically do:
    # preds = model.forward(batchX)
    # loss = criterion.forward(preds, batchY)
    # ...
next