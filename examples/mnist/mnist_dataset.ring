# File: examples/mnist_dataset.ring
# Description: Dataset loader for MNIST CSV format

class MnistDataset from Dataset
    aRawData
    nRows
    nClasses = 10 # Digits 0-9

    func init aData
        aRawData = aData
        nRows    = len(aRawData)

    func length
        return nRows

    func getData nIdx
        row = aRawData[nIdx]
        
        # --- 1. Process Label (First Column) ---
        nLabel = number(row[1]) # 0 to 9
        
        # One-Hot Encoding
        aOneHot = list(nClasses)
        for k=1 to nClasses aOneHot[k]=0 next
        
        # Ring lists are 1-based, so digit 0 goes to index 1, digit 9 to index 10
        aOneHot[nLabel + 1] = 1
        
        oTargetTensor = new Tensor(1, nClasses)
        oTargetTensor.aData[1] = aOneHot

        # --- 2. Process Image Pixels (Rest of Columns) ---
        nPixels = 784 # 28x28
        aPixels = list(nPixels)
        
        # CSV Row has 785 columns (1 label + 784 pixels)
        # Pixel 1 is at index 2
        for i = 1 to nPixels
            # Normalize 0-255 to 0.0-1.0
            val = number(row[i+1]) / 255.0
            aPixels[i] = val
        next
        
        oInTensor = new Tensor(1, nPixels)
        oInTensor.aData[1] = aPixels

        return [oInTensor, oTargetTensor]