from tkinter import *
from PIL import Image, ImageOps, ImageGrab
import numpy as np
import matplotlib.pyplot as plt

BRUSH_SIZE = 15
def clearCanvas():
    canvas.delete("all")

def predictCanvas():
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    x1, y1 = x + canvas.winfo_width(), y + canvas.winfo_height()
    image = ImageGrab.grab(bbox=(x, y, x1, y1)).crop()
    imageRes = image.resize((28, 28))
    imageArr = np.array(imageRes, dtype=np.uint8)[:, :, 0]

    _, img = plt.subplots(1, 2, figsize=(8, 4))

    img[0].imshow(image)
    img[0].set_title('Original Image')
    img[0].axis('off')

    img[1].imshow(imageRes)
    img[1].set_title('Resized Image')
    img[1].axis('off')

    plt.show()

    
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval((x - BRUSH_SIZE / 2, y - BRUSH_SIZE / 2, 
                        x + BRUSH_SIZE / 2, y + BRUSH_SIZE / 2), 
                       fill="white", outline="white")
    


window = Tk()
window.geometry("560x560")
window.resizable(False, False)

canvas = Canvas(window, width="280", height="280", bg="#000000", bd=0, highlightthickness=0)
canvas.place(relx=0.5, rely=0.5, anchor=CENTER)

predict = Button(window, text='Predict', command=predictCanvas, width=10)
predict.place(x=145, y=425)

clear = Button(window, text='Clear', command=clearCanvas, width=10)
clear.place(x=235, y=425)

exit = Button(window, text='Exit', command=quit, width=10)
exit.place(x=330, y=425)

res = Label(window, text="Result: ", width=32, relief=SUNKEN, anchor=W)
res.place(x=165, y=460)

canvas.bind("<B1-Motion>", draw)

window.mainloop()