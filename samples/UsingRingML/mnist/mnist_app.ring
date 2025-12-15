# File: examples/mnist_app.ring
# Description: Interactive MNIST Predictor with ASCII Art
# Author: Azzeddine Remmal

load "stdlib.ring"
load "ringml.ring"
load "mnist_dataset.ring"
load "csvlib.ring"

decimals(3)

see "=== RingML MNIST Digit Predictor ===" + nl

# 1. Load Data (For testing)
cFile = "data/mnist_test_lite.csv"
if !fexists(cFile) raise("File missing") ok

see "Loading Data..." + nl
aRawData = CSV2List( read(cFile) )
# if lower(aRawData[1][1]) = "label" del(aRawData, 1) ok

nTotal = len(aRawData)
see "Loaded " + nTotal + " test images." + nl

# 2. Build Model Structure (Must match training)
model = new Sequential
model.add(new Dense(784, 64))   
model.add(new ReLU)
# No need to add Dropout here, as loadWeights doesn't load dropout state anyway
# and we are in eval mode by default or explicit call.
model.add(new Dense(64, 32))  
model.add(new ReLU)
model.add(new Dense(32, 10)) 
model.add(new Softmax)

# 3. Load Weights
if !fexists("model/mnist_model.rdata")
    see "Error: Model file not found. Run mnist_train.ring first." + nl
    bye
ok
model.loadWeights("model/mnist_model.rdata")
model.evaluate() # Disable dropout logic if any

# 4. Interactive Loop
while true
    see nl + "Enter Image Index (1-" + nTotal + ") or 'exit': " 
    give cInput
    if lower(trim(cInput)) = "exit" exit ok
    
    nIdx = number(cInput)
    if nIdx < 1 or nIdx > nTotal 
        see "Invalid Index!" + nl 
        loop 
    ok
    
    # Get Data
    row = aRawData[nIdx]
    realLabel = number(row[1])
    
    # Create Input Tensor
    oInput = new Tensor(1, 784)
    for i = 1 to 784
        oInput.aData[1][i] = number(row[i+1]) / 255.0
    next
    
    # Predict
    pred = model.forward(oInput)
    
    # Find Max
    nMaxIdx = 0
    nMaxVal = -1
    for i = 1 to 10
        val = pred.aData[1][i]
        if val > nMaxVal
            nMaxVal = val
            nMaxIdx = i
        ok
    next
    
    predictedLabel = nMaxIdx - 1 # 1-based index to 0-9 digit
    
    # --- Visualization ---
    see nl + "Image " + nIdx + ":" + nl
    printAsciiArt(oInput)
    
    see "--------------------------" + nl
    see "Real Label: " + realLabel + nl
    see "AI Predict: " + predictedLabel 
    see " (Confidence: " + floor(nMaxVal*100) + "%)" + nl
    
    if realLabel = predictedLabel
        see "RESULT: CORRECT ✅" + nl
    else
        see "RESULT: WRONG ❌" + nl
    ok
end

func printAsciiArt oTensor
    # Draw 28x28 image using ASCII chars
    # Input is (1, 784) normalized 0-1
    
    cMap = " .:-=+*#%@"
    nLevels = len(cMap)
    
    idx = 1
    for y = 1 to 28
        line = ""
        for x = 1 to 28
            val = oTensor.aData[1][idx]
            # Map 0.0-1.0 to 1-10 char index
            charIdx = floor(val * (nLevels-1)) + 1
            line += cMap[charIdx] + " " # Extra space for aspect ratio
            idx++
        next
        see "    " + line + nl
    next