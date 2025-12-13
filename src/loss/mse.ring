# File: src/loss/mse.ring
# Description: Mean Squared Error Loss Function
# Author: Azzeddine Remmal


class MSELoss
    
    func forward oPred, oTarget
        oDiff = oPred.copy()
        oDiff.sub(oTarget)
        oDiff.square()
        return oDiff.mean()

    func backward oPred, oTarget
        # Gradient = 2 * (Pred - Target) / N
        
        oGrad = oPred.copy()
        oGrad.sub(oTarget)
        
        # DEBUG: Check sign of Pred - Target
        # see "   [MSE Debug] Diff (Pred - Target) [Should be negative]: " 
        # oGrad.print()
        
        oGrad.scalar_mul(2.0)
        
        nTotal = oPred.nRows * oPred.nCols
        oGrad.scalar_mul(1.0 / nTotal)
        
        return oGrad