from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod 
    def backward(self):
        pass