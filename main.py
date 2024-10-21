from tkinter import *
from PIL import Image, ImageGrab, ImageOps
from cnn import Conv, MaxPool, Softmax
import numpy as np
import matplotlib.pyplot as plt
import struct
import time

'''def loadImages(filename):
    with open(filename, 'rb') as file:
        magic, num, rows, cols = struct.unpack('>IIII', file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
        return images
    
def loadLabels(filename):
    with open(filename, 'rb') as file:
        magic, num = struct.unpack('>II', file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)
        return labels
    
trainImages = loadImages('train-images.idx3-ubyte')
trainLabels = loadLabels('train-labels.idx1-ubyte')
testImages = loadImages('t10k-images.idx3-ubyte')
testLabels = loadLabels('t10k-labels.idx1-ubyte')

conv = Conv(10)
pool = MaxPool()
sfmx = Softmax(13 * 13 * 10, 10)

def forward(image, label=None, predict=False):
    output = conv.forward((image / 255) - 0.5)
    output = pool.forward(output)
    output = sfmx.forward(output)

    if predict and not label:
        return np.argmax(output)
    
    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0
    
    return output, loss, acc


def training(image, label, learnRate=0.005):
    output, loss, acc = forward(image, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / output[label]

    gradient = sfmx.backprop(gradient, learnRate)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, learnRate)

    return loss, acc


print('MNIST CNN initialized!')
start = time.perf_counter()
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  permutation = np.random.permutation(len(trainImages))
  trainImages = trainImages[permutation]
  trainLabels = trainLabels[permutation]

  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(trainImages, trainLabels)):
    if i > 0 and i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        if num_correct == 100: break
        loss = 0
        num_correct = 0

    l, acc = training(im, label)
    loss += l
    num_correct += acc
end = time.perf_counter()
print(f'Training Runtime: {(end - start):.4f} seconds')

print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(testImages, testLabels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(testImages)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)'''

BRUSH_SIZE = 12
def draw(event):
    x = event.x
    y = event.y
    canvas.create_oval((x - BRUSH_SIZE / 2, y - BRUSH_SIZE / 2, 
                        x + BRUSH_SIZE / 2, y + BRUSH_SIZE / 2),
                        fill="white", outline="white" )

def clearCanvas():
    canvas.delete("all")
    res.set("Result: ")

def submitCanvas():
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab().crop((x, y, x1, y1))
    image = ImageOps.fit(image, (280, 280))
    resImage = image.resize((28, 28))
    imgArr = np.array(resImage, dtype=np.uint8)[:, :, 0]
    '''predict = forward(imgArr, predict=True)
    res.set(f"Result: {str(predict)}")'''
    
def training():
    pass

def startUp():
    win = Toplevel()
    win.title('Input Hyperparameters')
    win.geometry('380x80')

    epoch = Label(win, text='Epoch:', font=('Tahoma', 10, 'bold'))
    epoch.grid(row=0, column=0, padx=10, pady=8, sticky=W)

    epochInput = Entry(win, textvariable=epochs, font=('Tahoma', 10), relief=SUNKEN, width=10, justify=RIGHT)
    epochInput.grid(row=0, column=1)


    batch = Label(win, text='Batch Size:', font=('Tahoma', 10, 'bold'))
    batch.grid(row=1, column=0, padx=10, pady=8)
    

    batchInput = Entry(win, textvariable=batchSize, font=('Tahoma', 10), relief=SUNKEN, width=10, justify=RIGHT)
    batchInput.grid(row=1, column=1)

    learn = Label(win, text='Learn Rate:', font=('Tahoma', 10, 'bold'))
    learn.grid(row=0, column=2, padx=10, pady=8)
    
    learnInput = Entry(win, textvariable=learnRate, font=('Tahoma', 10), relief=SUNKEN, width=10, justify=RIGHT)
    learnInput.grid(row=0, column=3)

    startTrain = Button(win, text='Initialize', font=('Tahoma', 10), width=10)
    startTrain.grid(row=1, column=2, columnspan=2) 

window = Tk()
window.geometry("440x320")
window.title("OCR")
icon = PhotoImage(file='ocr.png')
window.iconphoto(True, icon)
window.resizable(False, False)

front = Frame(window)
back = Frame(window)

front.grid(row=0, column=0, sticky=NSEW)
back.grid(row=0, column=0, sticky=NSEW)

ocr = Label(front, text="Optical Character Recognition", font=('Tahoma', 19, 'bold'))
ocr.grid(row=0, column=0, padx=30, pady=8)

"This application leverages a Convolutional Neural Network (CNN) to predict handwritten digiys draw by the user on an interactive canvas."
desc = Label(front, text="This application leverages a Convolutional Neural Network (CNN) \nto predict handwritten digits drawn by the user on an interactive canvas.",
             font=('Tahoma', 10), justify='center')
desc.grid(row=1, column=0)

start = Button(front, text="start", width=10, command=startUp, font=('Tahoma', 10))
start.grid(row=3, column=0, pady=20)

###############################################################################################

canvas = Canvas(back, height="280", width="280", bg="#000000", bd=0, highlightthickness=0)
canvas.grid(row=0, column=0, padx=20, pady= 20, columnspan=4, rowspan=20)
canvas.bind('<B1-Motion>', draw)

acc = StringVar()
acc.set('Accuracy: ')
accuracy = Label(back, textvariable=acc, width=14, font=('Tahoma', 10), bd=3, anchor=SW)
accuracy.grid(row=2, column=4, columnspan=2, sticky=W)

submit = Button(back, text='submit', width=10, command=submitCanvas, font=('Tahoma', 10))
submit.grid(row=6, column=4, sticky=W)

train = Button(back, text='train', width=10, command=training, font=('Tahoma', 10))
train.grid(row=7, column=4, sticky=W)

res = StringVar()
res.set("Result: ") 
result = Label(back, textvariable=res, width=14, font=('Tahoma', 10), bd=3, anchor=W)
result.grid(row=8, column=4, columnspan=2, sticky=W)

clear = Button(back, text='clear', width=10, command=clearCanvas, font=('Tahoma', 10))
clear.grid(row=15, column=4, sticky=W)

exit = Button(back, text='exit', width=10, command=quit, font=('Tahoma', 10))
exit.grid(row=16, column=4, sticky=W)


front.tkraise()
window.mainloop()
