import numpy as np
        
class Conv:
    def __init__(self, numFilters):
        self.numfilters = numFilters
        self.filters = np.random.randn(numFilters, 3, 3) / 9

    def iterateRegion(self, image):
        imgHeight, imgWidth = image.shape
        for rows in range(imgHeight - 2):
            for cols in range(imgWidth - 2):
                imgReg = image[rows : (rows + 3), cols : (cols + 3)]
                yield imgReg, rows, cols

    def forward(self, input):
        self.cacheInput = input
        inHeight, inWidth = input.shape
        output = np.zeros((inHeight - 2, inWidth - 2, self.numfilters))
        for imgReg, rows, cols in self.iterateRegion(input):
            output[rows, cols] = np.sum(imgReg * self.filters, axis=(1, 2))
        return output

    def backprop(self, gradient, learnRate):
        dL_dfil = np.zeros(self.filters.shape)
        for imgReg, rows, cols in self.iterateRegion(self.cacheInput):
            for filt in range(self.numfilters):
                dL_dfil[filt] += gradient[rows, cols, filt] * imgReg
        self.filters -= learnRate * dL_dfil
        return None

class MaxPool:
    def iterateRegion(self, image):
        imgHeight, imgWidth, _ = image.shape
        for rows in range(imgHeight // 2):
            for cols in range(imgWidth // 2):
                imgReg = image[(rows * 2) : (rows * 2 + 2), (cols * 2) : (cols * 2 + 2)]
                yield imgReg, rows, cols
    
    def forward(self, input):
        self.cacheInput = input
        inHeight, inWidth, filters = input.shape
        output = np.zeros((inHeight // 2, inWidth // 2, filters))
        for imgReg, rows, cols in self.iterateRegion(input):
            output[rows, cols] = np.amax(imgReg, axis=(0, 1))
        return output

    def backprop(self, gradient):
        dL_din = np.zeros(self.cacheInput.shape)
        for imgReg, rows, cols in self.iterateRegion(self.cacheInput):
            amax = np.amax(imgReg, axis=(0, 1))
            for imgrRows in range(imgReg.shape[0]):
                for imgrCols in range(imgReg.shape[1]):
                    for imgrFilt in range(imgReg.shape[2]):
                        if imgReg[imgrRows, imgrCols, imgrFilt] == amax[imgrFilt]:
                            dL_din[rows * 2 + imgrRows, cols * 2 + imgrCols, imgrFilt] = gradient[rows, cols, imgrFilt]
        return dL_din
   
class Softmax:
    def __init__(self, inputLen, nodes):
        self.weight = np.random.randn(inputLen, nodes) / inputLen
        self.bias = np.zeros(nodes)

    def forward(self, input):
        self.cacheShape = input.shape
        input = input.flatten()
        self.cacheInput = input
        totals = np.dot(input, self.weight) + self.bias
        self.cacheTotal = totals
        exp = np.exp(totals)
        self.cacheExp = exp
        expSum = np.sum(exp, axis=0)
        self.cacheExpSum = expSum
        return exp / expSum

    def backprop(self, gradient, learnRate):
        for c, dL_dout in enumerate(gradient):
            if dL_dout == 0:
                continue

            exp = np.exp(self.cacheTotal)
            expSum = np.sum(exp)
            dout_dt = -exp[c] * exp / expSum ** 2
            dout_dt[c] = exp[c] * (expSum - exp[c]) / (expSum ** 2)

            dL_dt = dL_dout * dout_dt

            dt_dw = self.cacheInput
            dt_db = 1
            dt_din = self.weight

            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db =  dL_dt * dt_db
            dL_din = dt_din @ dL_dt

            self.weight -= learnRate * dL_dw
            self.bias -= learnRate * dL_db

            return dL_din.reshape(self.cacheShape)

