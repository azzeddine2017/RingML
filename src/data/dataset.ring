# File: src/data/dataset.ring
# Description: Dataset and DataLoader for batch processing
# Author: Code Gear-1


class Dataset
    # Base class for datasets
    func length 
        raise("Method length() not implemented")
    func getData itemIndex
        raise("Method getData() not implemented")

class TensorDataset from Dataset
    oInputs
    oTargets
    nSamples

    func init oInTensor, oTargetTensor
        oInputs  = oInTensor
        oTargets = oTargetTensor
        nSamples = oInputs.nRows
        
    func length
        return nSamples
        
    func getData nIdx
        # Returns a list [InputRow, TargetRow]
        # We need to extract specific row as a new Tensor (1, Cols)
        
        # Extract Input Row
        oInRow = new Tensor(1, oInputs.nCols)
        oInRow.aData[1] = oInputs.aData[nIdx] # Copy reference of the list
        
        # Extract Target Row
        oTargetRow = new Tensor(1, oTargets.nCols)
        oTargetRow.aData[1] = oTargets.aData[nIdx]
        
        return [oInRow, oTargetRow]

class DataLoader
    oDataset
    nBatchSize
    nBatches
    
    func init oData, nBatch
        oDataset   = oData
        nBatchSize = nBatch
        nTotal     = oDataset.length()
        nBatches   = ceil(nTotal / nBatchSize)

    func getBatch nBatchIndex
        # Calculate Start and End indices
        nStart = (nBatchIndex - 1) * nBatchSize + 1
        nEnd   = nStart + nBatchSize - 1
        
        if nEnd > oDataset.length()
            nEnd = oDataset.length()
        ok
        
        nCurrentBatchSize = nEnd - nStart + 1
        if nCurrentBatchSize <= 0 return [] ok
        
        # Peek first item to know dimensions
        firstItem = oDataset.getData(1)
        nFeats  = firstItem[1].nCols
        nLabels = firstItem[2].nCols
        
        # Allocate Batch Tensors ONCE
        oBatchInputs  = new Tensor(nCurrentBatchSize, nFeats)
        oBatchTargets = new Tensor(nCurrentBatchSize, nLabels)
        
        # Fill Batch
        rowCounter = 1
        for i = nStart to nEnd
            item = oDataset.getData(i) # Lazy load row i
            
            # Copy data from small tensor to batch tensor
            oBatchInputs.aData[rowCounter]  = item[1].aData[1]
            oBatchTargets.aData[rowCounter] = item[2].aData[1]
            
            rowCounter++
        next
        
        return [oBatchInputs, oBatchTargets]