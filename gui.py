import tkinter as tk
from tkinter import filedialog, Text, Label, RAISED
import os
import cv2
from PIL import Image,ImageTk
import featureBased

featureBasedClassifier = featureBased.FeatureBasedClassifier();
featureBasedClassifier.loadImageClasses()

def loadImage(canvas, img_id):
    global imgtk
    global imageToClassifyPure
    global imageToDisplay
    filename = filedialog.askopenfilename(initialdir="/",title="Select Image",
                                          filetypes=[('image files','*.png')])
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
    cv2.putText(imageToDisplay,result.resultString,(20,50),cv2.QT_FONT_NORMAL,1,(255,0,10),2)
    image = cv2.drawMatches(result.img1,result.kp1,imageToClassifyPure,result.kp2,result.matches,None)
    dim = (1000,550)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    im = Image.fromarray(image)
    imgtk2 = ImageTk.PhotoImage(image=im)
    canvas.itemconfig(img_id, image=imgtk2)
    match_label.config(text="Best match: " + result.accuracy + " times the standard deviation")
    # canvas.itemconfig(img_right_id, image=imgtk2)



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
classifyImageCnn= tk.Button(root, text="Classify Image CNN",padx=10,pady=5, fg="black")
classifyImageCnn.pack()

root.mainloop()