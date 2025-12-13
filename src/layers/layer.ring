# File: src/layers/layer.ring
# Description: Base abstract class (Updated with bTrainable)
# Author: Code Gear-1

class Layer
    # Mode Flag: True = Training, False = Inference/Testing
    bTraining = true 
    
    # Freeze Flag: If false, weights won't update during training
    bTrainable = true

    func train
        bTraining = true
        
    func evaluate
        bTraining = false
        
    # New: Freeze layer weights
    func freeze
        bTrainable = false
        
    # New: Unfreeze layer weights
    func unfreeze
        bTrainable = true

    func forward oInputTensor
        raise("Error: forward() not implemented")
        
    func backward oGradientTensor
        raise("Error: backward() not implemented")