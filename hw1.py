import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
import numpy as np 
import cv2
from PIL import Image, ImageTk, ImageEnhance
import math
import matplotlib.pyplot as plt

win = Tk()
win.geometry('1200x1200')
win.title('GUI')
chosen = None
maxa = 0

def select_file():#開啟檔案
    global lab1, lab2, chosen, img, img2, im
    chosen = fd.askopenfilename()
    img = cv2.imread(chosen) #用opencv來處理
    img2 = cv2.imread(chosen)

    im = Image.fromarray(img) #將陣列轉換為圖片
    im = im.resize((250, 250), Image.ANTIALIAS) #用pillow調整圖片大小,因為woman圖片原本太大
    imgtk = ImageTk.PhotoImage(im)
    lab1.configure(image = imgtk)
    lab1.img = imgtk

    im = Image.fromarray(img2)
    im = im.resize((250, 250), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(im)
    lab2.configure(image = imgtk)
    lab2.img = imgtk

def save_file():#存檔
    filename = fd.asksaveasfile(mode = 'wb', filetypes = [("All files", "*.*"), ("JPG files", "*.jpg"), ("TIF files", "*.tif")])
    if not filename:#若上一行有錯則返回
        return
    
    if chosen:#已經有開檔
        Image.fromarray(img).save(filename)

counter = 0
def changeMethod():#決定現在市linear,Exponential,或是log
    global counter
    counter = counter + 1
    
    if counter % 3 == 0:
        aslider.set(100)
        bslider.set(0)
        method['text'] = 'Linear'
        aslider['to_'] = 600
        bslider['from_'] = -100
        bslider['to_'] = 100  
    elif counter % 3 == 1:
        aslider.set(1)
        bslider.set(0)
        method['text'] = 'Expo'
        aslider['to_'] = 125
        bslider['from_'] = -16
        bslider['to_'] = 5
    else:
        aslider.set(1)
        bslider.set(1)
        method['text'] = 'Log'
        aslider['to_'] = 1e35
        bslider['from_'] = 1
        bslider['to_'] = 1e35

def adjcontrastbright(x):
    global img, img2 

    a = aslider.get()
    b = bslider.get()
    for y in range(img.shape[0]):#照片為3d
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                if counter % 3 == 0:
                    tmp = 0.01 * a * img2[y, x, c] + b
                elif counter % 3 == 1:
                    tmp = np.exp(0.001 * a * img2[y, x, c] + b)
                else:
                    tmp = np.log(a * float(img2[y, x, c]) + b)

                if tmp > 255:#pixel大於255就用255存
                    img[y, x, c] = 255
                elif tmp < 0:
                    img[y, x, c] = 0
                else:
                    img[y, x, c] = tmp


    im = Image.fromarray(img) #將陣列轉換為圖片
    im = im.resize((250, 250), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(im)
    lab1.configure(image = imgtk)
    lab1.img = imgtk

def zoomig(x):
    global img, chosen
    if chosen:
        img = cv2.imread(chosen)
        s = zoombar.get()
        s *= 0.1
        img = cv2.resize(img, None, fx = s, fy = s, interpolation = cv2.INTER_LINEAR)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(im)
        lab1.configure(image = imgtk)
        lab1.img = imgtk


def rotate(x):#旋轉圖片
    global img, img2, im
    hei, wid = img2.shape[:2] #讀取影像的長和高
    R = cv2.getRotationMatrix2D((wid/2, hei/2), degree.get(), 1) #旋轉影像矩陣,選擇以中心為中心點,圖片大小固定
    img = cv2.warpAffine(img2, R, (wid, hei))
    im = Image.fromarray(img) #將陣列轉換為圖片
    im = im.resize((250, 250), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(im)
    lab1.configure(image = imgtk)
    lab1.img = imgtk

def showhist():
    global img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()
    plt.hist(gray, bins = 256)
    plt.show()

def histequal():
    global img
    im = Image.fromarray(img)
    imgray = im.convert(mode = 'L') #轉成灰階
    im_arr = np.asarray(imgray) #轉成numpy array
    hist_arr = np.bincount(im_arr.flatten(), minlength = 256) #計算每個灰階出現幾次
    #標準化
    num_pix = np.sum(hist_arr)
    hist_arr = hist_arr / num_pix
    culhist_arr = np.cumsum(hist_arr)
    trans_map = np.floor(255 * culhist_arr).astype(np.uint8)

    img_list = list(im_arr.flatten())
    eq_img_list = [trans_map[p] for p in img_list] #均值化
    eq_img_arr = np.reshape(np.asarray(eq_img_list), im_arr.shape)
    
    eq_img = Image.fromarray(eq_img_arr, mode = 'L')
    im = eq_img.resize((250, 250), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(im)
    lab1.configure(image = imgtk)
    lab1.img = imgtk
    

open_button = tk.Button(win, text = 'Open files', command = select_file)
save_button = tk.Button(win, text = 'Save files', command = save_file)
lab1 = Label(win, image = "", width = 250, height = 250)
lab2 = Label(win, image = "")
method = tk.Button(win, text = 'Linear', command = changeMethod) #只要按按紐計算的方式就換轉換
aslider = tk.Scale(win, from_ = 1, to_ = 600, label = 'contrast', length = 300, orient = 'horizontal', command = adjcontrastbright)
aslider.set(100)
bslider = tk.Scale(win, from_ = -100, to_ = 100, label = 'bright', length = 300, orient = 'horizontal', command = adjcontrastbright)
zoombar = tk.Scale(win, from_ = 1, to_ = 100, label = 'zoom', length = 300, orient = 'horizontal', command = zoomig)
zoombar.set(10)
degree = tk.Scale(win, from_ = 0, to_ = 360, label = 'degree', length = 300, orient = 'horizontal', command = rotate)
showbut = tk.Button(win, text = 'show histogram', command = showhist)
histeq = tk.Button(win, text = 'histogram equalization', command = histequal)

open_button.place(x = 0, y = 100)
save_button.place(x = 0, y = 150)
lab1.place(x = 200, y = 50)
lab2.place(x = 600, y = 50)
method.place(x = 0, y = 300)
aslider.place(x = 200, y = 300)
bslider.place(x = 550, y = 300)
zoombar.place(x = 200, y = 450)
degree.place(x = 200, y = 600)
showbut.place(x = 0, y = 700)
histeq.place(x = 0, y = 750)

win.mainloop()