from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageGrab, ImageOps
from cnn import Conv, MaxPool, Softmax
import numpy as np
import matplotlib.pyplot as plt
import struct
import time


def loadImages(filename):
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

def training(image, label, learnRate):
    output, loss, acc = forward(image, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / output[label]

    gradient = sfmx.backprop(gradient, learnRate)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, learnRate)

    return loss, acc

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

def predictCanvas():
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab().crop((x, y, x1, y1))
    image = ImageOps.fit(image, (280, 280))
    resImage = image.resize((28, 28))
    imgArr = np.array(resImage, dtype=np.uint8)[:, :, 0]
    predict = forward(imgArr, predict=True)
    res.set(f"Result: {str(predict)}")

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

    startTrain = Button(win, text='Initialize', font=('Tahoma', 10), width=10, 
                        command=lambda: (win.withdraw(), 
                                         initTrain(int(epochInput.get()), 
                                                   float(learnInput.get()), 
                                                   int(batchInput.get())
                                                   )))
    startTrain.grid(row=1, column=2, columnspan=2)

def initTrain(epoch, learnRate, batchSize):
    global front
    trainImages = loadImages('train-images.idx3-ubyte')[:batchSize]
    trainLabels = loadLabels('train-labels.idx1-ubyte')[:batchSize]
    testImages = loadImages('t10k-images.idx3-ubyte')[:batchSize]
    testLabels = loadLabels('t10k-labels.idx1-ubyte')[:batchSize]
    Label(front, text='MNIST Initialized!', font=('Tahoma', 12)).grid(row=5, column=0, columnspan=2, pady=10, padx=15)
    front.update()

    start = time.perf_counter()
    for i in range(epoch):
        print('--- Epoch %d ---' % (i + 1))

        permutation = np.random.permutation(len(trainImages))
        trainImages = trainImages[permutation]
        trainLabels = trainLabels[permutation]

        message = Label(front, text=f'Training Phase: Epoch {i + 1}', font=('Tahoma', 10))
        message.grid(row=6, column=0, pady=5, padx=15, sticky=W)
        bar = Progressbar(front, orient=HORIZONTAL, length=400, maximum=(batchSize // 100), mode='determinate')
        bar.grid(row=7, column=0, columnspan=2, pady=5, padx=15)
        step = Label(front)
        front.update()

        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(trainImages, trainLabels)):
            if i > 0 and i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                bar['value'] += 1
                #Label(front, text=f'{i // 30}%', font=('Tahoma', 10)).grid(row=7, column=0, columnspan=2, pady=5, padx=15)
                step.config(text=f'[Step {i + 1}] Past 100 steps: Average Loss {loss / 100: .3f} | Accuracy: {num_correct}%',  font=('Tahoma', 10))
                step.grid(row=8, column=0, columnspan=2, pady=5, padx=15)
                front.update()
                loss = 0
                num_correct = 0

            l, acc = training(im, label, learnRate)
            loss += l
            num_correct += acc

    end = time.perf_counter()
    print(f'Training Runtime: {(end - start):.4f} seconds')

    time.sleep(1)

    message.config(text='Testing Phase')
    step.config(text='')
    front.update()
    loss = 0
    num_correct = 0
    for im, label in zip(testImages, testLabels):
        _, l, acc = forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(testImages)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)

    messagebox.showinfo(title='Done', message=f'Done Training with accuracy of {(num_correct / num_tests) * 100: .2f}% \n Elapsed time: {(end - start):.2f} seconds')
    accu.set(f'Accuracy:{(num_correct / num_tests) * 100: .1f}%')
    back.tkraise()

def trainCanvas(label, learnRate):
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab().crop((x, y, x1, y1))
    image = ImageOps.fit(image, (280, 280))
    resImage = image.resize((28, 28))
    imgArr = np.array(resImage, dtype=np.uint8)[:, :, 0]
    _, _ = training(imgArr, label, learnRate)
    messagebox.showinfo(title='Success', message='Training Success')
    
def inputLabel():
    win = Toplevel()
    win.title('Label')
    win.geometry('300x50')
    win.resizable(False, False)

    label = Label(win, text='Label Value:', font=('Tahoma', 10))
    label.grid(row=0, column=0, padx=10, pady=10, sticky=W)

    labelInput = Entry(win, textvariable=labelValue, font=('Tahoma', 10), relief=SUNKEN, width=10, justify=RIGHT)
    labelInput.grid(row=0, column=1, sticky=W)

    submit = Button(win, text='Submit', command=lambda: (trainCanvas(int(labelInput.get()), int(learnRate.get())), win.withdraw()), font=('Tahoma', 10), width=10)
    submit.grid(row=0, column=2, columnspan=2, padx=30)

window = Tk()
window.geometry("450x320")
window.title("OCR")
icon = PhotoImage(file='ocr.png')
window.iconphoto(True, icon)
window.resizable(False, False)

front = Frame(window)
back = Frame(window)

front.grid(row=0, column=0, sticky=NSEW)
back.grid(row=0, column=0, sticky=NSEW)

ocr = Label(front, text="Optical Character Recognition", font=('Tahoma', 19, 'bold'))
ocr.grid(row=0, column=0, columnspan=2, padx=30, pady=8)

"This application leverages a Convolutional Neural Network (CNN) to predict handwritten digiys draw by the user on an interactive canvas."
desc = Label(front, text="This application leverages a Convolutional Neural Network (CNN) \nto predict handwritten digits drawn by the user on an interactive canvas.",
             font=('Tahoma', 10), justify='center')
desc.grid(row=1, column=0, columnspan=2)

start = Button(front, text="start", width=10, command=startUp, font=('Tahoma', 10))
start.grid(row=3, column=0, columnspan=2, pady=20)

###############################################################################################

canvas = Canvas(back, height="280", width="280", bg="#000000", bd=0, highlightthickness=0)
canvas.grid(row=0, column=0, padx=20, pady= 20, columnspan=4, rowspan=20)
canvas.bind('<B1-Motion>', draw)

accu = StringVar()
accu.set('Accuracy: ')
accuracy = Label(back, textvariable=accu, width=14, font=('Tahoma', 10), bd=3, anchor=SW)
accuracy.grid(row=2, column=4, columnspan=2, sticky=W)

predict = Button(back, text='Predict', width=10, command=predictCanvas, font=('Tahoma', 10))
predict.grid(row=6, column=4, sticky=W)

train = Button(back, text='Train', width=10, command=inputLabel, font=('Tahoma', 10))
train.grid(row=7, column=4, sticky=W)

res = StringVar()
res.set("Result: ") 
result = Label(back, textvariable=res, width=14, font=('Tahoma', 10), bd=3, anchor=W)
result.grid(row=8, column=4, columnspan=2, sticky=W)

clear = Button(back, text='Clear', width=10, command=clearCanvas, font=('Tahoma', 10))
clear.grid(row=15, column=4, sticky=W)

exit = Button(back, text='Exit', width=10, command=quit, font=('Tahoma', 10))
exit.grid(row=16, column=4, sticky=W)

###############################################################################################

epochs = IntVar()
epochs.set(3)
batchSize = IntVar()
batchSize.set(1000)
learnRate = IntVar()
learnRate.set(0.005)
labelValue = IntVar()

###############################################################################################

front.tkraise() 
window.mainloop()
