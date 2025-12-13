# File: examples/chess_train_batch.ring
# Description: Efficient Batch Training for Chess KRK

load "../../src/ringml.ring"
load "chess_utils.ring"
load "chess_dataset.ring" # Load our custom dataset
load "csvlib.ring"
load "stdlib.ring"
decimals(3)

see "=== RingML Chess Training (Mini-Batch) ===" + nl

# 1. Load CSV (Raw List only - Fast)
cFile = "data/chess.csv"

see "Reading CSV..." + nl
aRData = CSV2List( read(cFile) )



if len(aRData) > 0 del(aRData, 1) ok # Remove Header

nRow = len(aRData)
see "Loaded " + nRow + " games." + nl

# 2. Setup Dataset & DataLoader
# We do NOT create huge tensors here. We just wrap the list.
oDataset = new ChessDataset(aRData)

# Batch Size 64 is good balance between speed and memory
batch_size = 64
oLoader = new DataLoader(oDataset, batch_size)

see "Created DataLoader: " + oLoader.nBatches + " batches per epoch." + nl

# 3. Model
nClasses = 18
oModel = new Sequential
oModel.add(new Dense(6, 64))   
oModel.add(new Sigmoid)        
oModel.add(new Dense(64, 32))  
oModel.add(new Sigmoid)
oModel.add(new Dense(32, nClasses)) 
oModel.add(new Softmax)

oModel.summary()

# 4. Train
oCriterion = new CrossEntropyLoss
oOptimizer = new SGD(0.1) 
nEpochs   = 50 

see "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    epochLoss = 0
    
    # Iterate over Batches
    for b = 1 to oLoader.nBatches
        # This creates Tensors for only 64 rows! (Very Fast)
        batch = oLoader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        # Forward
        preds = oModel.forward(inputs)
        loss  = oCriterion.forward(preds, targets)
        epochLoss += loss
        
        # Backward
        grad = oCriterion.backward(preds, targets)
        oModel.backward(grad)
        
        # Update
        for layer in oModel.getLayers() oOptimizer.update(layer) next
        
        # Optional: Print progress within epoch
        if b % 100 = 0 see "." ok
    next
    
    avgLoss = epochLoss / oLoader.nBatches
    see "Epoch " + epoch + " Avg Loss: " + avgLoss + nl
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

# 5. Save
oModel.saveWeights("model/chess_model.rdata")
see "Model Saved." + nl