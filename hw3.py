import tkinter
from tkinter import *
from tkinter import filedialog as fd
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageFilter
import math
import matplotlib.pyplot as plt

win = Tk()
win.geometry('1280x960')
win.title('GUI')
photowidth = 0
photoheight = 0

def showimg1(img):
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    lab1.configure(image = imgtk)
    lab1.img = imgtk

def showimg2(img):
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    lab2.configure(image = imgtk)
    lab2.img = imgtk

def whiteBar():
    global photowidth, photoheight
    chosen = fd.askopenfilename() #please chose 'BarTest.tig'
    if chosen:
        load = Image.open(chosen)
        sample = photo = np.array(load)
        showimg1(sample)

        option = varFilter.get()
        if option[0] =='7':
            if option[4] == 'a':
                mask = np.ones((7, 7), np.float32)
                mask = mask / 49
                tmp = cv2.filter2D(photo, -1, mask)
            else:
                tmp = cv2.medianBlur(photo, 7)
        else:
            if option[4] == 'a':
                mask = np.ones((3, 3), np.float32)
                mask = mask / 9
                tmp = cv2.filter2D(photo, -1, mask)
            else:
                tmp = cv2.medianBlur(photo, 3)
        output = tmp.astype(np.uint8)
        showimg2(output)

def colorImage():
    global lenna
    lenna  = fd.askopenfilename() #pleas chose 'Lenna_512_color.tif'
    if lenna:
        load = Image.open(lenna)
        sample = photo = np.array(load)
        showimg1(sample)
        showimg2(photo)

def rgbComponent():
    global lenna
    load = Image.open(lenna)
    data = load.getdata()
    option = varColor.get()
    if option[0] == 'R':
        channel = [(d[0], 0, 0) for d in data]
    elif option[0] == 'G':
        channel = [(0, d[1], 0) for d in data]
    else:
        channel = [(0, 0, d[2]) for d in data]

    img = load
    img.putdata(channel)
    img = np.array(img)
    showimg2(img)

def rgbtohsi():
    global lenna
    load = Image.open(lenna)
    rgb = cv2.cvtColor(np.array(load), cv2.COLOR_BGR2RGB)
    #to[0, 1]
    bgr = np.float32(rgb) / 255.0
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    H = np.copy(r)

    for i in range(0, load.size[0]):
        for j in range(0, load.size[1]):
            #calculate theta
            x = 0.5 * ( (r[i][j]-g[i][j]) + (r[i][j]-b[i][j]))
            y = math.sqrt( (r[i][j]-g[i][j])**2  + ((r[i][j]-b[i][j])*(g[i][j]-b[i][j])) )
            z = math.acos(x/y)
            #decide h by b and g
            if b[i][j] < g[i][j]:
                H[i][j] = z
            else:
                H[i][j] = ((360*math.pi)/180.0) - z
    #calculate s
    min = np.minimum(np.minimum(r,g),b)
    S = 1 - (3 / (r+g+b+0.001) * min)
    #calculate i
    I = np.divide(b+g+r, 3.0)

    plt.subplot(1,3,1),plt.imshow(H,cmap='gray'),plt.title('Hue')
    plt.subplot(1,3,2),plt.imshow(S,cmap='gray'),plt.title('Saturation')
    plt.subplot(1,3,3),plt.imshow(I,cmap='gray'),plt.title('Intensity')
    plt.show()

def colorComplement():
    global lenna
    load = Image.open(lenna)
    data = load.getdata()
    #complement
    complement = [(255-d[0], 255-d[1], 255-d[2]) for d in data]
    img = load
    img.putdata(complement)
    img = np.array(img)
    showimg2(img)

def smoothing():
    global lenna
    option = varModel.get()
    load = Image.open(lenna)
    if option[0] == 'R':
        mask = np.ones((5, 5), np.float32)
        mask = mask / 25
        tmp = cv2.filter2D(np.array(load), -1, mask)
        output = tmp.astype(np.uint8)
        showimg2(output)
    if option[0] == 'H':
        img = cv2.cvtColor(np.array(load), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img)
        mask = np.ones((5, 5), np.float32)
        mask = mask / 25
        tmp = cv2.filter2D(np.array(v), -1, mask)
        v = tmp.astype(np.uint8)
        img = cv2.merge([h, s, v])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        showimg2(img)

def sharpening():
    global lenna
    option = varModel.get()
    load = Image.open(lenna)
    if option[0] == 'R':
        sharp = load.filter(ImageFilter.Kernel((5, 5), (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)))
        showimg2(np.array(sharp))
    else:
        img = cv2.cvtColor(np.array(load), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img)
        mask = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 25, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])
        tmp = cv2.filter2D(np.array(v), -1, mask)
        v = tmp.astype(np.uint8)
        img = cv2.merge([h, s, v])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        showimg2(img)

def segment():#use hue and saturation to segment feathers
    global lenna
    output = load = Image.open(lenna)
    hue = Image.open(lenna)
    sat = Image.open(lenna)
    for i in range(512):
        for j in range(512):
                R, G, B = load.getpixel((i, j))
                if i > 50 and i < 300 and j > 160:
                    h = math.acos((((R-G)+(R-B))/2)/(math.sqrt(math.pow((R-G), 2)+(R-B)*(G-B))))*180/math.pi
                    if B > G:
                        h = int(255-(h/360)*255)
                    else:
                        h = int((h/360)*255)

                    s = 255-int(255*(1-(3/(R+G+B))*min(R, G, B)))
                else:
                    h = 0
                    s = 0
                
                if h < 190 or h > 230:
                    h = 0
                if s < 80 or s > 240:
                    s = 0
                
                if s == 0 or h == 0:
                    R = G = B = 0
                
                hue.putpixel((i, j), (h, 0, 0))
                sat.putpixel((i, j), (0, s, 0))
                output.putpixel((i, j), (R, G, B))
    showimg2(np.array(output))
    allimg = np.hstack((np.array(hue), np.array(sat), np.array(output)))
    cv2.imshow('hue, saturation, final', allimg)

choseFilter = ['7x7 arithmetic mean filter', '3x3 arithmetic mean filter', '7x7 median filter', '3x3 median filter']
varFilter = StringVar()
varFilter.set(choseFilter[0])
filterOption = OptionMenu(win, varFilter, *choseFilter)
filterOption.place(x = 0, y = 0)
showWhiteBar = Button(win, text = 'show white bar', command = whiteBar)
showWhiteBar.place(x = 0, y = 30)
lab1 = Label(win, image = "", width = photowidth, height = photoheight)
lab1.place(x = 180, y = 0)
lab2 = Label(win, image = "", width = photowidth, height = photoheight)
lab2.place(x = 700, y = 0)

colorImg = Button(win, text = 'show Lenna', command = colorImage)
colorImg.place(x = 0, y = 60)
choseRGB = ['Red component image', 'Green component image', 'Blue component image']
varColor = StringVar()
varColor.set(choseRGB[0])
colorOption = OptionMenu(win, varColor, *choseRGB)
colorOption.place(x = 0, y = 90)
showRGB = Button(win, text = 'show rgb image', command = rgbComponent)
showRGB.place(x = 0, y = 120)
hsi = Button(win, text = 'RGB to HSI', command = rgbtohsi)
hsi.place(x = 0, y = 150)
complementRGB = Button(win, text = 'color complement', command = colorComplement)
complementRGB.place(x = 0, y = 180)
choseModel = ['RGB', 'HSI']
varModel = StringVar()
varModel.set(choseModel[0])
modelOption = OptionMenu(win, varModel, *choseModel)
modelOption.place(x = 0, y = 210)
smooth = Button(win, text = 'smoothing', command = smoothing)
smooth.place(x = 0, y = 240)
sharpen = Button(win, text = 'sharpening', command = sharpening)
sharpen.place(x = 0, y = 270)
seg = Button(win, text = 'segment feather', command = segment)
seg.place(x = 0, y = 300)
mainloop()