# File: src/layers/softmax.ring
# Description: Softmax Layer for Classification
# Author: Azzeddine Remmal



class Softmax from Layer
    oOutput

    func forward oInputTensor
        oOutput = oInputTensor.copy()
        oOutput.softmax()
        return oOutput

    func backward oGradOutput
        # In Deep Learning, Softmax is usually combined with CrossEntropy.
        # The complex derivative cancels out nicely to (Pred - Target).
        # So we pass the gradient through, letting CrossEntropy handle the math.
        return oGradOutput