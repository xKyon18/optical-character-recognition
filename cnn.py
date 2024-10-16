import numpy as np


class Conv:
    def __init__(self, numFilters):
        self.numFilters = numFilters
        self.filters = np.random.randn(numFilters, 3, 3) / 9

    def iterateRegion(self, image):
        height, width = image.shape
        for rows in range(height - 2):
            for cols in range(width - 2):
                imgReg = image[rows : rows + 3, cols : cols + 3]
                yield imgReg, rows, cols

    def forward(self, input):
        height, width = input.shape
        output = np.zeros((height - 2, width - 2, self.numFilters))
        for imgReg, rows, cols in self.iterateRegion(input):
            output[rows, cols] = np.sum(imgReg * self.filters, axis=(1, 2))
        return output
    
    def backprop(self, gradient):
        pass

class MaxPool:
    def iterateRegion(self, image):
        height, width, _ = image.shape
        for rows in range(height // 2):
            for cols in range(width // 2):
                imgReg = image[rows * 2 : (rows * 2) + 2, cols * 2: (cols * 2) + 2]
                yield imgReg, rows, cols
    
    def forward(self, input):
        height, width, filters = input.shape
        output = np.zeros((height // 2, width // 2, filters))
        for imgReg, rows, cols in self.iterateRegion(input):
            output[rows, cols] = np.amax(imgReg, axis=(0,1))
        return output
    
    def backprop(self, gradient):
        pass

class Softmax:
    def __init__(self, inputLen, nodes):
        self.weight = np.random.randn(inputLen, nodes) / inputLen
        self.bias = np.zeros(nodes)

    def forward(self, input):
        input = input.flatten()
        total = input @ self.weight + self.bias
        exp = np.exp(total)
        return exp / (np.sum(exp))
    
    def backprop(self, gradient):
        pass
