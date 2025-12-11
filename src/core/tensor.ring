# File: src/core/tensor.ring
# Description: Core Tensor class - High Performance Mode (FastPro Fixed)
# Author: Code Gear-1

load "fastpro.ring"

class Tensor
    aData   = []    # Matrix data
    nRows   = 0     # Number of rows
    nCols   = 0     # Number of columns

    func init nR, nC
        nRows = nR
        nCols = nC
        aData = list(nRows)
        for i = 1 to nRows
            aData[i] = list(nCols)
        next
        return self
        
    func copy
        oNew = new Tensor(nRows, nCols)
        for r = 1 to nRows
            oNew.aData[r] = aData[r] 
        next
        return oNew

    # --- Initialization ---

    func random
        updateList(aData, :random, :matrix)
        return self

    func zeros
        aData = updateList(aData, :zerolike, :matrix)
        return self

    func fill nVal
        aData = updateList(aData, :fill, :matrix, nVal)
        return self

    # --- Math Operations ---

    func add oTensor
        checkDimensions(oTensor)
        aData = updateList(aData, :add, :matrix, oTensor.aData)
        return self

    func sub oTensor
        checkDimensions(oTensor)
        aData = updateList(aData, :sub, :matrix, oTensor.aData)
        return self

    func mul oTensor
        checkDimensions(oTensor)
        # Element-Wise Mul (Using the new emul command if available, or manual loop)
        # Fallback to manual loop if emul not yet compiled in your DLL
        for r = 1 to nRows
             for c = 1 to nCols
                 aData[r][c] *= oTensor.aData[r][c]
             next
        next
        return self

    func scalar_mul nVal
        aData = updateList(aData, :scalar, :matrix, nVal)
        return self

    func sum nAxis
        newData = updateList(aData, :sum, :matrix, nAxis)
        if nAxis = 1 
            oRes = new Tensor(nRows, 1)
        else
            oRes = new Tensor(1, nCols)
        ok
        oRes.aData = newData
        return oRes
        
    func square
        aData = updateList(aData, :square, :matrix)
        return self
        
    func mean
        nRes = updateList(aData, :mean, :matrix)
        return nRes

    func exp_func
        # Requires FastPro update or manual loop. 
        # Manual loop for safety:
        for r = 1 to nRows
            for c = 1 to nCols
                aData[r][c] = exp(aData[r][c])
            next
        next
        return self

    # --- Matrix Operations ---

    func matmul oTensor
        if nCols != oTensor.nRows
            raise("Dimension Mismatch in MatMul")
        ok
        newData = updateList(aData, :mul, :matrix, oTensor.aData)
        oRes = new Tensor(nRows, oTensor.nCols)
        oRes.aData = newData
        return oRes

    func transpose
        aData = updateList(aData, :transpose, :matrix)
        nTemp = nRows
        nRows = nCols
        nCols = nTemp
        return self

    # --- Activations ---

    func sigmoid
        aData = updateList(aData, :sigmoid, :matrix)
        return self

    func sigmoid_prime
        aData = updateList(aData, :sigmoidprime, :matrix)
        return self

    func relu
        aData = updateList(aData, :relu, :matrix)
        return self
        
    func relu_prime
        aData = updateList(aData, :reluprime, :matrix)
        return self
        
    func tanh
        aData = updateList(aData, :tanh, :matrix)
        return self

    func softmax
        # FastPro Case 3306 (Now Implemented in C)
        aData = updateList(aData, :softmax, :matrix)
        return self
        
    # --- Utilities ---

    func checkDimensions oTensor
        if nRows != oTensor.nRows or nCols != oTensor.nCols
            raise("Dimension Mismatch")
        ok

    func print
        see "Tensor Shape: (" + nRows + ", " + nCols + ")" + nl
        if nRows <= 10 and nCols <= 10
            see aData
        else
            see "Data is too large to display." + nl
        ok
        see nl