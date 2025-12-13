# File: examples/chess_app.ring
# Description: Console App to predict Chess End-Game result

load "../../src/ringml.ring"
load "chess_utils.ring"


see "==========================================" + nl
see "      Chess End-Game Predictor (AI)       " + nl
see "==========================================" + nl

# 1. Build Model Architecture (Must match training)
model = new Sequential
model.add(new Dense(6, 32))   
model.add(new Sigmoid)        
model.add(new Dense(32, 16))  
model.add(new Sigmoid)
model.add(new Dense(16, 18)) 
model.add(new Softmax)

# 2. Load Weights
if !fexists("model/chess_model_fast.rdata")
    see "Error: Model file 'chess_model_fast.rdata' not found." + nl
    see "Please run chess_train_fast.ring first." + nl
    bye
ok

model.loadWeights("model/chess_model_fast.rdata")
see "Model loaded successfully." + nl + nl

# 3. User Input Loop
while true
    see "------------------------------------------" + nl
    see "Enter Board Configuration (or 'exit'):" + nl
    
    see "White King File (a-h): " give wk_f
    if lower(trim(wk_f)) = "exit" exit ok
    
    see "White King Rank (1-8): " give wk_r
    see "White Rook File (a-h): " give wr_f
    see "White Rook Rank (1-8): " give wr_r
    see "Black King File (a-h): " give bk_f
    see "Black King Rank (1-8): " give bk_r
    
    try
        # Process Input
        nWK_F = getFileIndex(wk_f)
        nWK_R = number(wk_r)
        nWR_F = getFileIndex(wr_f)
        nWR_R = number(wr_r)
        nBK_F = getFileIndex(bk_f)
        nBK_R = number(bk_r)
        
        # Validation
        if nWK_F<1 or nWK_F>8 or nWK_R<1 or nWK_R>8 raise("Invalid Coord") ok
        
        # Normalize
        normData = normalizeBoard([nWK_F, nWK_R, nWR_F, nWR_R, nBK_F, nBK_R])
        
        # Prepare Tensor
        inputTensor = new Tensor(1, 6)
        inputTensor.aData[1] = normData
        
        # Predict
        prediction = model.forward(inputTensor)
        
        # Find ArgMax (Class with highest probability)
        nMaxIdx = 0
        nMaxVal = -1
        
        # prediction is (1, 18)
        for i = 1 to 18
            val = prediction.aData[1][i]
            if val > nMaxVal
                nMaxVal = val
                nMaxIdx = i
            ok
        next
        
        resultStr = getLabelName(nMaxIdx)
        
        see nl + ">>> AI Prediction: " + resultStr 
        see " (Confidence: " + floor(nMaxVal*100) + "%)" + nl
        
    catch
        see "Invalid Input! Please try again." + nl
    done
end