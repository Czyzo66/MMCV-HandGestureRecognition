import tkinter as tk
from tkinter import filedialog, Text, Label, RAISED
import os
import cv2
from PIL import Image,ImageTk
import featureBasedSift
import featureBasedOrb
import cnnClassifier
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

featureBasedClassifier = featureBasedSift.FeatureBasedClassifier();
featureBasedClassifier.loadImageClasses()
featureBasedClassifierOrb = featureBasedOrb.FeatureBasedClassifierOrb();
featureBasedClassifierOrb.loadImageClasses()
cnnClassifier = cnnClassifier.CnnClassifier()

detailsClasses = []
detailsMatches = []


def loadImage(canvas, img_id):
    global imgtk
    global imageToClassifyPure
    global imageToDisplay
    filename = filedialog.askopenfilename(initialdir="C:/Users/mitux/Desktop/woda/data/train",title="Select Image",
                                          filetypes=[('image files','*.png',)])
    imageToClassifyPure = cv2.imread(filename)
    imageToClassify = imageToClassifyPure
    # cv2.imshow('test',imageToClassify)

    b,g,r = cv2.split(imageToClassify)
    imageToClassify = cv2.merge((r,g,b))
    dim = (500,550)
    imageToClassify = cv2.resize(imageToClassify, dim, interpolation = cv2.INTER_LINEAR)
    imageToDisplay =imageToClassify

    im = Image.fromarray(imageToClassify)

    # im = im.resize((500,550), Image.ANTIALIAS)

    imgtk = ImageTk.PhotoImage(image=im)
    canvas.itemconfig(img_id, image=imgtk)
    print(imgtk)

def classifySift(canvas, img_id, img_right_id, match_label):

    result = featureBasedClassifier.classify(imageToClassifyPure)
    global imgtk2
    cv2.putText(imageToDisplay,result.resultString,(20,50),cv2.QT_FONT_NORMAL,.5,(255,0,10),1)
    image = cv2.drawMatches(result.img1,result.kp1,imageToClassifyPure,result.kp2,result.matches,None)
    dim = (1000,550)
    cv2.putText(image,"original",(300,30),cv2.QT_FONT_NORMAL,.9,(255,0,10),1)
    cv2.putText(image,"match",(30,30),cv2.QT_FONT_NORMAL,.9,(255,0,10),1)

    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    im = Image.fromarray(image)
    imgtk2 = ImageTk.PhotoImage(image=im)
    canvas.itemconfig(img_id, image=imgtk2)
    match_label.config(text="Best match: " + result.resultString +", Confidence: " + result.accuracy + " times the standard deviation")
    # canvas.itemconfig(img_right_id, image=imgtk2)
    global detailsClasses
    detailsClasses= result.classNames
    global detailsMatches
    detailsMatches= result.goodMatchList

def classifyOrb(canvas, img_id, match_label):
    result = featureBasedClassifierOrb.classify(imageToClassifyPure)
    global imgtk2
    image = cv2.drawMatches(result.img1,result.kp1,imageToClassifyPure,result.kp2,result.matches,None)
    dim = (1000,550)
    cv2.putText(image,"original",(300,30),cv2.QT_FONT_NORMAL,.9,(255,0,10),1)
    cv2.putText(image,"match",(30,30),cv2.QT_FONT_NORMAL,.9,(255,0,10),1)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

    im = Image.fromarray(image)
    imgtk2 = ImageTk.PhotoImage(image=im)
    canvas.itemconfig(img_id, image=imgtk2)
    match_label.config(text="Best match: "+ result.resultString +", Confidence: " + result.accuracy + " times the standard deviation")
    global detailsClasses
    detailsClasses= result.classNames
    global detailsMatches
    detailsMatches= result.goodMatchList

    # canvas.itemconfig(img_right_id, image=imgtk2)

def classifyCnn(canvas, img_id, match_label):
    result = cnnClassifier.classify(imageToClassifyPure)
    global imgtk2
    # image = cv2.drawMatches(result.img1,result.kp1,imageToClassifyPure,result.kp2,result.matches,None)
    dim = (500,550)
    # image = cv2.resize(result.img, dim, interpolation=cv2.INTER_LINEAR)
    imggg = cv2.resize(imageToClassifyPure,dim,interpolation=cv2.INTER_LINEAR, dst=imageToClassifyPure)
    cv2.putText(imggg,result.resultString,(50,50),cv2.QT_FONT_NORMAL,.9,(255,0,10),2)
    # im = Image.fromarray(imageToClassifyPure)
    im = Image.fromarray(imggg)
    imgtk2 = ImageTk.PhotoImage(image=im)
    canvas.itemconfig(img_id, image=imgtk2)
    # match_label.config(text="Best match: "+ result.resultString +", Confidence: " + result.accuracy + " times the standard deviation")
    # global detailsClasses
    # detailsClasses= result.classNames
    # global detailsMatches
    # detailsMatches= result.goodMatchList

    # canvas.itemconfig(img_right_id, image=imgtk2)

def openSecondWindow():
    top = tk.Toplevel()
    top.title('Detailed results')
    details_plot_title_label = tk.Label(top,text="Number of matches per class").pack()
    fig = Figure(figsize=(5, 4), dpi=100)
    # ax = fig.add_axes([0,0,1,1])
    ax = fig.add_subplot(111)
    ax.bar(detailsClasses,detailsMatches[0],.5)
    # fig.add_subplot(111).(detailsClasses,detailsMatches[0])
    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # toolbar = NavigationToolbar2Tk(canvas, )
    # toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    details_classes_label = tk.Label(top,text=str(detailsClasses)).pack()
    details_matches_label = tk.Label(top,text=str(detailsMatches)).pack()
    btn = tk.Button(top,text="close window",command=top.destroy, padx=10,pady=5,fg="black").pack()



root = tk.Tk()
canvas = tk.Canvas(root, height=600, width=1000)
canvas.pack()
img_id = canvas.create_image((0,0), image=None, anchor='nw')
img_right_id= canvas.create_image((500,0), image=None, anchor='nw')


# frame = tk.Frame(root, bg="black")
# frame.place(relwidth=0.8, relheight=0.8, relx=0.1)
my_label = tk.Label(root, text = "Results:")
match_label = tk.Label(root, text = "Best match: ")

my_label.pack()
match_label.pack()


openFile = tk.Button(root,text="Load Image", padx=10,pady=5, fg="black",
                     command=lambda: loadImage(canvas,img_id))
openFile.pack();

classifyImage= tk.Button(root, text="Classify Image SIFT",padx=10,pady=5, fg="black",
                         command=lambda: classifySift(canvas,img_id, img_right_id,match_label))
classifyImage.pack()
classifyImageOrb = tk.Button(root, text="Classify Image ORB", padx=10,pady=5, fg="black",
                             command=lambda: classifyOrb(canvas,img_id,match_label))
classifyImageOrb.pack()

classifyImageCnn= tk.Button(root, text="Classify Image CNN",padx=10,pady=5, fg="black",
                            command=lambda: classifyCnn(canvas,img_id,match_label))
classifyImageCnn.pack()

seconWindowBtn = tk.Button(root,text="Show detailed results",padx=10,pady=5,fg="black", command=openSecondWindow).pack()

root.mainloop()