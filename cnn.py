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
        self.cacheInput = input
        height, width, filters = input.shape
        output = np.zeros((height // 2, width // 2, filters))
        for imgReg, rows, cols in self.iterateRegion(input):
            output[rows, cols] = np.amax(imgReg, axis=(0,1))
        return output
    
    def backprop(self, gradient):
        dL_din = np.zeros((self.cacheInput.shape))
        for imgReg, rows, cols in self.iterateRegion(self.cacheInput):
            amax = np.amax(imgReg, axis=(0, 1)
            for imgrRows in range(imgReg.shape[0]):
                for imgrCols in range(imgReg.shape[1]):
                    for imgrFilt in range(imgReg.shape[2]):
                        if imgReg[imgrRows, imgrCols, imgrFilters] == amax[imgrFilt]:
                            dL_din[(rows * 2) + imgrRows, (cols * 2) + imgrCols, imgrFilt] = gradient[rows, cols, imgrFilt]
        return dL_din
        
class Softmax:
    def __init__(self, inputLen, nodes):
        self.weight = np.random.randn(inputLen, nodes) / inputLen
        self.bias = np.zeros(nodes)

    def forward(self, input):
        self.cacheShape = input.shape
        input = input.flatten()
        self.cacheInput = input
        total = input @ self.weight + self.bias
        self.cacheTotal = total
        exp = np.exp(total)
        return exp / (np.sum(exp))
    
    def backprop(self, gradient, learnRate):
        for c, dL_dout in enumerate(gradient):
            if dL_dout == 0:
                continue
            exp = np.exp(self.cacheTotal)
            expSum = np.sum(exp)
            dout_dt = np.zeros((self.cacheTotal.shape))
            dout_dt[c] = (exp[c] * (expSum - exp)) / expSum ** 2

            dL_dt = dL_dout * dout_dt #(10,)

            dt_dw = self.cacheInput #(1690,)
            dt_db = 1
            dt_din = self.weight #(1690, 10)

            dL_dw = dt_din[np.newaxis].T @ dl_dt[np.newaxis]
            dL_db = dL_dt
            dL_din = dt_din @ dL_dt

            self.weight -= learnRate * dL_dw
            self.bias -= learnRate * dL_db

            return dL_din.reshape(self.cacheShape)
            
            
