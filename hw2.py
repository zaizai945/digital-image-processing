import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
import numpy as np 
import cv2
from PIL import Image, ImageTk, ImageEnhance
import math
import matplotlib.pyplot as plt

win = Tk()
win.geometry('1500x1500')
win.title('GUI')

def showimg(img):
    im = Image.fromarray(img)
    im = im.resize((512, 512), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(im)
    lab1.configure(image = imgtk)
    lab1.img = imgtk

def show(img):
    im = Image.fromarray(img)
    im = im.resize((512, 512), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(im)
    lab2.configure(image = imgtk)
    lab2.img = imgtk

def select_file():#開啟檔案
    global lab1, chosen, load, output, printer
    chosen = fd.askopenfilename()
    if chosen:
        load = Image.open(chosen)
        printer = output = np.array(load)
        render = load.resize((512, 512), Image.ANTIALIAS) #用pillow調整圖片大小,因為woman圖片原本太大
        imgtk = ImageTk.PhotoImage(render)
        lab1.configure(image = imgtk)
        lab1.img = imgtk

def preserve_slicing(x):
    global output, load, chosen, printer
    output = np.array(load)
    minimum = pmins.get()
    maximum = pmaxs.get()
    if(maximum < minimum):
        maximum, minimum = minimum, maximum
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j] > minimum and output[i, j] < maximum: #pixel's value is in the interval then be 255. By the way interval could be (-1, 256) for both slicing
                output[i, j] = 255
    printer = output
    show(printer)

def discard_slicing(x):
    global output, load, printer
    output = np.array(load)
    minimum = dmins.get()
    maximum = dmaxs.get()
    if(maximum < minimum):
        maximum, minimum = minimum, maximum
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j] > minimum and output[i, j] < maximum:
                output[i, j] = 255
            else:
                output[i, j] = 0
    printer = output
    show(printer)

#In bit() my idead was trans decimal to binary and use scale to decide which plan should i take
#If the pixel value on the plane is '1' then make it to 255 otherwise to 0 casue it's kind of slicing
def bit(x):
    global output, load, printer
    output = np.array(load)
    plane = bs.get()
    reg = output.tolist()
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            tmp = [0, 0, 0, 0, 0, 0, 0, 0]
            for k in range(8):
                tmp[k] = reg[i][j] % 2
                reg[i][j] = reg[i][j] // 2
            reg[i][j] = tmp[plane]
    output = np.asarray(reg)
    output[output > 0] = 255
    output = np.uint8(output)
    printer = output
    show(printer)

def aversmoothing(x):
    global output, load, printer
    val = avsmooth.get()
    output = np.array(load)
    img = output
    mask = np.ones([val, val], dtype = int)
    mask = mask / (val**2)
    tmp = cv2.filter2D(img, -1, mask)
    output = tmp.astype(np.uint8)
    printer = output
    show(printer)

def gausmoothing(x):
    global output, chosen, printer
    val = gusmooth.get()
    output = cv2.imread(chosen)
    output = cv2.GaussianBlur(output, (val, val), 0)
    printer = output
    show(printer)

def mediansmoothing(x):
    global output, chosen, printer
    val = medsmooth.get()
    output = cv2.imread(chosen)
    output = cv2.medianBlur(output, val)
    printer = output
    show(printer)

 #the scale value will decide the kernel size and the value must be odd.
 #if scale vale is greater, the center value will become greater than before so it will still focus on the specific pixel
def sharpening(x):
    global load, output, printer
    size = sharpen.get()
    if size % 2:
        output = np.array(load)
        kernel = [[-1] * size for i in range(size)]
        kernel[size//2][size//2] = size**2
        kernel = np.asarray(kernel)
        output = cv2.filter2D(output, -1, kernel)
        printer = output
        show(printer)

def pirateopen():#please only opne pirate file
    global photo, rawfile, chosen
    rawfile = fd.askopenfilename()
    if rawfile:
        chosen = True
        f = open(rawfile, 'rb+')
        tmp = np.fromfile(f, dtype = np.uint8, count = -1)
        photo = tmp.reshape((512, 512))
        showimg(photo)
        f.close()

def pirateav():
    global photo, output, printer
    img = photo
    mask = np.ones([3, 3], dtype = int)
    mask = mask / 9
    tmp = cv2.filter2D(img, -1, mask)
    output = tmp.astype(np.uint8)
    printer = output
    show(printer)

def piratemedian():
    global photo, output, printer
    img = photo
    newimg = np.zeros([512, 512])
    for i in range(1, 511):
        for j in range(1, 511):
            tmp = [img[i-1, j-1], img[i-1, j], img[i-1, j + 1], img[i, j-1], img[i, j], img[i, j + 1], img[i + 1, j-1], img[i + 1, j], img[i + 1, j + 1]]
            tmp = sorted(tmp)
            newimg[i, j] = tmp[4]
    output = newimg.astype(np.uint8)
    printer = output
    show(printer)

def bestimp():
    global photo, output, rawfile, printer
    if rawfile[-5] == 'a':
        tmp = cv2.medianBlur(photo, 3)
    else:
        tmp = cv2.blur(photo, (3, 3))
    output = cv2.Laplacian(tmp, cv2.CV_16S, 3)
    printer = output
    show(printer)

def denoising():
    global output, printer
    printer = cv2.fastNlMeansDenoising(output, None, 7, 7, 21)
    show(printer)

def save_file():#存檔
    global printer, chosen
    filename = fd.asksaveasfile(mode = 'wb', filetypes = [("All files", "*.*"), ("TIF files", "*.tif")])
    if not filename:#若上一行有錯則返回
        return
    
    if chosen:#已經有開檔
        Image.fromarray(printer).save(filename)

open_button = tk.Button(win, text = 'Open', command = select_file)
save_button = tk.Button(win, text = 'Save', command = save_file)
lab1 = Label(win, image = "", width = 512, height = 512)
lab2 = Label(win, image = "", width = 512, height = 512)
pmins = tk.Scale(win, from_ = -1, to_ = 256, label = 'preserve slice a', length = 300, orient = 'horizontal', command = preserve_slicing)
pmaxs = tk.Scale(win, from_ = -1, to_ = 256, label = 'preserve slice b', length = 300, orient = 'horizontal', command = preserve_slicing)
dmins = tk.Scale(win, from_ = -1, to_ = 256, label = 'unselected to zero a', length = 300, orient = 'horizontal', command = discard_slicing)
dmaxs = tk.Scale(win, from_ = -1, to_ = 256, label = 'unselected to zero b', length = 300, orient = 'horizontal', command = discard_slicing)
bs = tk.Scale(win, from_ = 0, to_ = 7, label = 'bit plane', length = 200, orient = 'horizontal', command = bit)
avsmooth = tk.Scale(win, from_ = 3, to_ = 40, label = 'average mask', length = 100, orient = 'horizontal', command = aversmoothing)
gusmooth = tk.Scale(win, from_ = 3, to_ = 40, label = 'gaussian mask', length = 100, orient = 'horizontal', command = gausmoothing)
medsmooth = tk.Scale(win, from_ = 3, to_ = 40, label = 'median mask', length = 100, orient = 'horizontal', command = mediansmoothing)
sharpen = tk.Scale(win, from_ = 2, to_ = 9, label = 'sharpening', length = 100, orient = 'horizontal', command = sharpening)
pirate = tk.Button(win, text = 'open raw pirate', command = pirateopen)
rawav = tk.Button(win, text = 'raw file average mask', command = pirateav)
rawmed = tk.Button(win, text = 'raw file median mask', command = piratemedian)
rawbest = tk.Button(win, text = 'best improve and Laplace', command = bestimp)
denois = tk.Button(win, text = 'denois', command = denoising)

open_button.place(x = 0, y = 100)
save_button.place(x = 0, y = 150)
pirate.place(x = 0, y = 200)
rawav.place(x = 0, y = 550)
rawmed.place(x = 0, y = 600)
rawbest.place(x = 0, y = 650)
lab1.place(x = 130, y = 0)
lab2.place(x = 650, y = 0)
pmins.place(x = 200, y = 550)
pmaxs.place(x = 550, y = 550)
dmins.place(x = 200, y = 620)
dmaxs.place(x = 550, y = 620)
bs.place(x = 200, y = 700)
avsmooth.place(x = 900, y = 550)
gusmooth.place(x = 900, y = 620)
medsmooth.place(x = 900, y = 700)
sharpen.place(x = 1010, y = 550)
denois.place(x = 0, y = 700)

win.mainloop()
