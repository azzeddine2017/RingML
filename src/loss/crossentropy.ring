# File: src/loss/crossentropy.ring
# Description: Cross Entropy Loss
# Author: Azzeddine Remmal


class CrossEntropyLoss
    
    func forward oPred, oTarget
        # Loss = -Sum(Target * log(Pred)) / BatchSize
        
        nTotalLoss = 0
        nBatch = oPred.nRows
        nClasses = oPred.nCols
        
        for r = 1 to nBatch
            for c = 1 to nClasses
                pred = oPred.aData[r][c]
                target = oTarget.aData[r][c]
                npred = 0.000000000000001
                # Clip prediction to avoid log(0) error
                if pred < npred pred = npred ok
                if pred > (1.0 - npred) pred = 1.0 - npred ok
                
                # Assuming One-Hot Encoded target
                if target = 1
                    nTotalLoss -= log(pred)
                ok
            next
        next
        
        return nTotalLoss / nBatch

    func backward oPred, oTarget
        # Combined derivative of Softmax + CrossEntropy is:
        # Gradient = Pred - Target
        
        oGrad = oPred.copy()
        oGrad.sub(oTarget)
        
        # Normalize by batch size
        nTotal = oPred.nRows 
        oGrad.scalar_mul(1.0 / nTotal)
        
        return oGrad