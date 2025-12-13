# File: src/optim/sgd.ring
# Description: Stochastic Gradient Descent Optimizer (Fixed)
# Author: Code Gear-1

class SGD
    nLearningRate = 0.01

    func init nLR
        nLearningRate = nLR

    func update oLayer
        # --- FIX: Check if layer is trainable ---
        if hasAttribute(oLayer, "bTrainable") 
            if !oLayer.bTrainable return ok
        ok
        # Check if layer has weights/bias using proper reflection
        if variableExists(oLayer, "oWeights") and variableExists(oLayer, "oGradWeights")
            
            # Calculate Step: lr * dW
            oStepW = oLayer.oGradWeights.copy()
            oStepW.scalar_mul(nLearningRate)
            
            # Apply Step: W = W - Step
            oLayer.oWeights.sub(oStepW)
        ok

        # Update Bias: B = B - (lr * dB)
        if variableExists(oLayer, "oBias") and variableExists(oLayer, "oGradBias")
            
            oStepB = oLayer.oGradBias.copy()
            oStepB.scalar_mul(nLearningRate)
            
            oLayer.oBias.sub(oStepB)
        ok
        
    func variableExists oObj, cVar
        # Correct way to check attributes in Ring
        aAttrs = attributes(oObj)
        for cAttr in aAttrs
            # Compare case-insensitive
            if lower(cAttr) = lower(cVar) 
                return true 
            ok
        next
        return false

    