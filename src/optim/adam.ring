# File: src/optim/adam.ring
# Description: Adam Optimizer (Manual Implementation for Stability)
# Author: Code Gear-1

class Adam
    nLR = 0.001
    nBeta1 = 0.9
    nBeta2 = 0.999
    nEpsilon = 0.00000001
    
    func init nLearningRate
        nLR = nLearningRate

    func update oLayer
        # Check trainability
        if hasAttribute(oLayer, "bTrainable") 
            if !oLayer.bTrainable return ok
        ok

        # Check weights
        if !hasAttribute(oLayer, "oWeights") return ok

        # --- Initialize State ---
        if !hasAttribute(oLayer, "adam_mw")
            # State for Weights
            addAttribute(oLayer, "adam_mw")
            addAttribute(oLayer, "adam_vw")
            oLayer.adam_mw = oLayer.oWeights.copy().zeros()
            oLayer.adam_vw = oLayer.oWeights.copy().zeros()
            
            # State for Bias
            addAttribute(oLayer, "adam_mb")
            addAttribute(oLayer, "adam_vb")
            oLayer.adam_mb = oLayer.oBias.copy().zeros()
            oLayer.adam_vb = oLayer.oBias.copy().zeros()
            
            addAttribute(oLayer, "adam_t")
            oLayer.adam_t = 0
        ok

        # Increment Time Step
        oLayer.adam_t++
        nT = oLayer.adam_t

        # Update Weights (Manual Loop)
        update_param_manual(
            oLayer.oWeights, 
            oLayer.oGradWeights, 
            oLayer.adam_mw, 
            oLayer.adam_vw, 
            nT
        )

        # Update Bias (Manual Loop)
        update_param_manual(
            oLayer.oBias, 
            oLayer.oGradBias, 
            oLayer.adam_mb, 
            oLayer.adam_vb, 
            nT
        )

    func update_param_manual oParam, oGrad, oM, oV, nT
        # Calculate corrections once
        correction1 = 1.0 - pow(nBeta1, nT)
        correction2 = 1.0 - pow(nBeta2, nT)
        
        nRows = oParam.nRows
        nCols = oParam.nCols
        
        # Optimization: Direct access to lists
        aW = oParam.aData
        aG = oGrad.aData
        aM = oM.aData
        aV = oV.aData
        
        for r = 1 to nRows
            for c = 1 to nCols
                g = aG[r][c]
                
                # 1. Update biased first moment estimate
                # m = beta1 * m + (1 - beta1) * g
                aM[r][c] = (nBeta1 * aM[r][c]) + ((1.0 - nBeta1) * g)
                
                # 2. Update biased second raw moment estimate
                # v = beta2 * v + (1 - beta2) * g^2
                aV[r][c] = (nBeta2 * aV[r][c]) + ((1.0 - nBeta2) * (g * g))
                
                # 3. Bias Correction
                m_hat = aM[r][c] / correction1
                v_hat = aV[r][c] / correction2
                
                # 4. Update Parameter
                # W = W - (lr * m_hat) / (sqrt(v_hat) + epsilon)
                
                # Safety against sqrt of negative (rare float error)
                if v_hat < 0 v_hat = 0 ok
                
                nDenom = sqrt(v_hat) + nEpsilon
                nStep = (nLR * m_hat) / nDenom
                
                aW[r][c] -= nStep
            next
        next

    