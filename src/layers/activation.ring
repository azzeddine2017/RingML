# File: src/layers/activation.ring
# Description: Activation Layers (ReLU, Sigmoid) with Backprop
# Author: Code Gear-1



class Activation from Layer
    oOutput # Cache output

class Sigmoid from Activation
    func forward oInputTensor
        # Cache output for backward
        oOutput = oInputTensor.copy()
        oOutput.sigmoid()
        return oOutput

    func backward oGradOutput
        # dInput = dOutput * Sigmoid'(Output)
        # S'(x) = x * (1-x) calculated by :sigmoidprime
        
        oDerivative = oOutput.copy()
        oDerivative.sigmoid_prime() 
        
        # Element-wise multiplication
        oGradOutput.mul(oDerivative)
        
        return oGradOutput

class ReLU from Activation
    oInputCache 
    
    func forward oInputTensor
        oInputCache = oInputTensor.copy()
        
        oOutput = oInputTensor.copy()
        oOutput.relu()
        return oOutput

    func backward oGradOutput
        # dInput = dOutput * ReLU'(Input)
        
        oDerivative = oInputCache.copy()
        oDerivative.relu_prime() 
        
        oGradOutput.mul(oDerivative)
        return oGradOutput

class Tanh from Activation
    oOutput

    func forward oInputTensor
        # Cache output for backward pass
        oOutput = oInputTensor.copy()
        oOutput.tanh()
        return oOutput

    func backward oGradOutput
        # Gradient = GradOutput * (1 - Output^2)
        oDerivative = oOutput.copy()
        oDerivative.tanh_prime()
        
        oGradOutput.mul(oDerivative)
        return oGradOutput