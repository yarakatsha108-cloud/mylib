from Layers.Base import Base
import numpy as np

class Dense_Affine(Base):
    def __init__ (self, input_size, output_size, weight_init="he" , b = None):
        super().__init__()
        if weight_init == "he":
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif weight_init == "glorot":
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01
        if b is None:
            self.b = np.zeros(output_size)
        else:
            self.b = b
            
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) 
        if self.b is not None:
            out += self.b
        return out
    
    def backward(self, doubt):
        dx = np.dot(doubt, self.W.T)
        self.dW = np.dot(self.x.T, doubt)
        if self.b is not None:
            self.db = np.sum(doubt, axis=0)  
        return dx