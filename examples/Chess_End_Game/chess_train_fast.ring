# File: examples/chess_train_fast.ring
# Description: Lightweight Training for slower machines

load "stdlib.ring"
load "../../src/ringml.ring"
load "chess_utils.ring"
load "chess_dataset.ring"
load "csvlib.ring"

# Set display precision to 5 decimal places
decimals(8)

see "=== RingML Chess Training (Fast Mode) ===" + nl

# 1. Load Data
cFile = "data/chess.csv"

see "Reading CSV..." + nl
aRawData = CSV2List( read(cFile) )
if len(aRawData) > 0 del(aRawData, 1) ok 

nRow = len(aRawData)
see "Loaded " + nRow + " games." + nl

# 2. Setup Dataset 
dataset = new ChessDataset(aRawData)

# OPTIMIZATION 1: Larger Batch Size
# 256 means fewer updates per epoch = Faster processing
batch_size = 256 
loader = new DataLoader(dataset, batch_size)

see "Batches per epoch: " + loader.nBatches + nl

# 3. Model (Lightweight Architecture)
nClasses = 18
model = new Sequential

# OPTIMIZATION 2: Smaller Hidden Layers (32 instead of 64)
# Input(6) -> Dense(32) -> Sigmoid -> Dense(16) -> Sigmoid -> Output(18)
model.add(new Dense(6, 32))   
model.add(new Sigmoid)        
model.add(new Dense(32, 16))  
model.add(new Sigmoid)
model.add(new Dense(16, nClasses)) 
model.add(new Softmax)

model.summary()

# 4. Train
criterion = new CrossEntropyLoss
optimizer = new SGD(0.2) # Slightly higher LR for larger batches
nEpochs   = 1 # Start with 50 to test speed

see "Starting Training..." + nl
tTotal = clock()

for epoch = 1 to nEpochs
    epochLoss = 0
    
    for b = 1 to loader.nBatches
        batch = loader.getBatch(b) 
        inputs  = batch[1]
        targets = batch[2]
        
        preds = model.forward(inputs)
        loss  = criterion.forward(preds, targets)
        epochLoss += loss
        
        grad = criterion.backward(preds, targets)
        model.backward(grad)
        
        for layer in model.getLayers() optimizer.update(layer) next

        if b % 5 = 0 see "." ok

        # OPTIMIZATION 3: Force Garbage Collection every few batches
        if b % 50 = 0  callgc() ok
    next
    see nl
    avgLoss = epochLoss / loader.nBatches
    see "Epoch " + epoch + " Avg Loss: " + avgLoss + nl
next

see "Total Time: " + ((clock()-tTotal)/clockspersecond()) + "s" + nl

model.saveWeights("model/chess_model_fast.rdata")
see "Model Saved." + nl