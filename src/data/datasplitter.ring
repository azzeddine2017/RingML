# File: src/data/datasplitter.ring
# Description: Utility class for Shuffling and Splitting Data
# Author: Code Gear-1

class DataSplitter

    # Function: train_test_split
    # Input: Raw Data List, Test Ratio (e.g. 0.2), Shuffle (True/False)
    # Output: List [TrainList, TestList]
    func splitData aData, nTestRatio, bShuffle
        
        nTotal = len(aData)
        if nTotal = 0 raise("Error: Data is empty") ok

        # 1. Shuffle Data (In-Place) if requested
        if bShuffle
            shuffle(aData)
        ok

        # 2. Calculate Split Index
        nTestSize = floor(nTotal * nTestRatio)
        nTrainSize = nTotal - nTestSize
        
        aTrain = []
        aTest  = []

        # 3. Distribute Data
        # Optimization: We loop through references
        for i = 1 to nTotal
            if i <= nTrainSize
                aTrain + aData[i]
            else
                aTest + aData[i]
            ok
        next

        return [aTrain, aTest]

    # Fisher-Yates Shuffle Algorithm
    func shuffle aList
        nLen = len(aList)
        for i = nLen to 2 step -1
            j = random(i-1) + 1
            temp = aList[i]
            aList[i] = aList[j]
            aList[j] = temp
        next
        return aList