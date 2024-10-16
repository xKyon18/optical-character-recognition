from cnn import Conv, MaxPool, Softmax
import numpy as np 
import struct

def loadImages(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
        return images
    
def loadLabels(filename):
    with open(filename, 'rb') as file:
        magic, num, = struct.unpack('>II', file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)
        return labels
    
trainImages = loadImages('train-images.idx3-ubyte')[:200]
trainLabels = loadLabels('train-labels.idx1-ubyte')[:200]
testImages = loadImages('t10k-images.idx3-ubyte')
testLabels = loadLabels('t10k-labels.idx1-ubyte')

conv = Conv(10)
pool = MaxPool()
sfmx = Softmax(13 * 13 * 10, 10)

def forward(image, label, predict=False):
    output = conv.forward((image / 255) - 0.5)
    output = pool.forward(output)
    output = sfmx.forward(output)

    if predict:
        return np.argmax(output)
    
    loss = -np.log(output)
    acc = 1 if np.argmax(output) == label else 0
    
    return output, loss, acc


def training(image, label, learnRate):
    output, loss, acc = forward(image, label)

    gradient = np.zeros(output)
    gradient[label] = -1 / output[label]

    gradient = sfmx.backprop(gradient, learnRate)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, learnRate)

    return loss, acc

print("Initialize Training")

loss, acc = 0, 0
for i, (image, label) in enumerate(zip(trainImages, trainLabels)):
    if i % 19 == 0:
        print("|\tEpoch: %d \t|", i + 1)
        print("Loss: %.3f, Accuracy: %d", loss / 20, acc)
        loss = 0
        acc = 0

    l, a = training(image, label)
    loss += l
    acc += a


