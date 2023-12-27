import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import CNNModel 
#from tkvideo import tkvideo

#import tfModel_test as tf_test
global fn
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Home Page")




# #####For background Image
image2 = Image.open('11.jpg')
image2 = image2.resize((1290, 790), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=250, y=1)  # , relwidth=1, relheight=1)





lbl = tk.Label(root, text="Dibetes Mellitus Detection System", font=('times', 30,' bold '), width=65,height=2,bg="black",fg="white")
lbl.place(x=0, y=0)

# video_label =tk.Label(root)
# video_label.pack()

# # read video to display on label

# player = tkvideo("cell3.gif", video_label,loop = 1, size = (w, h))
# player.play()


#frame_display = tk.LabelFrame(root, text=" --Display-- ", width=900, height=250, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display.grid(row=0, column=0, sticky='nw')
#frame_display.place(x=300, y=100)

#frame_display1 = tk.LabelFrame(root, text=" --Result-- ", width=900, height=200, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display1.grid(row=0, column=0, sticky='nw')
#frame_display1.place(x=300, y=430)

#frame_display2 = tk.LabelFrame(root, text=" --Calaries-- ", width=900, height=50, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display2.grid(row=0, column=0, sticky='nw')
#frame_display2.place(x=300, y=380)


# wlcm=tk.Label(root,text="......Welcome Brain_Stroke_CT-SCAN_image detection System......",width=100,height=1,background="#FF6103",foreground="white",font=("Times new roman",19,"bold"))
# wlcm.place(x=0,y=655)


frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=262, height=760, bd=5, font=('times', 14, ' bold '),bg="midnightblue")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=2, y=94)


    
###########################################################################
def train_model():
 
    update_label("Model Training Start...............")
    
    start = time.time()

    X= CNNModel.main()
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    print(msg)

import functools
import operator


def convert_str_to_tuple(tup):
    s = functools.reduce(operator.add, (tup))
    return s

def test_model_proc(fn):
    from keras.models import load_model
#    from keras.optimizers import Adam

#    global fn
    
    IMAGE_SIZE = 64
    LEARN_RATE = 1.0e-4
    CH=3
    print(fn)
    if fn!="":
        # Model Architecture and Compilation
       
        #model = load_model('ocular_disease.h5')
            
        # adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        # Load the model
        model = load_model("E:/Dibetes Mellitus/model1.h5", compile=False)
        
        # Load the labels
        class_names = open("labels.txt", "r").readlines()
        
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = np.array(img)
        
        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
        
        img = img.astype('float32')
        img = img / 255.0
        print('img shape:',img)
        # prediction = model.predict(img)
        # print(prediction)





        # print(np.argmax(prediction))
        # brain=np.argmax(prediction)
        # print(brain)
        
        
        # # Create the array of the right shape to feed into the keras model
        # # The 'length' or number of images you can put into the array is
        # # determined by the first position in the shape tuple, in this case 1
        # data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        # # Replace this with the path to your image
        # image = Image.open("<IMAGE_PATH>").convert("RGB")
        
        # # resizing the image to be at least 224x224 and then cropping from the center
        # size = (224, 224)
        # image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # # turn the image into a numpy array
        # image_array = np.asarray(image)
        
        # # Normalize the image
        # normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # # Load the image into the array
        # data[0] = normalized_image_array
        
        # Predicts the model
        prediction = model.predict(img)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        # print("Confidence Score:", confidence_score)
        
        
        
        if index == 0:
            Cd="Control Group"
        else:
            Cd="DM Group"
        
       
        A=Cd
        return A

# def clear_img():
    
#     img11 = tk.Label(frame_display, background='lightblue4',width=160,height=120)
#     img11.place(x=0, y=0)

def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='goldenrod', fg='black')
    result_label.place(x=300, y=450)
# def train_model():
    
#     update_label("Model Training Start...............")
    
#     start = time.time()

#     X=Model_frm.main()
    
#     end = time.time()
        
#     ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
#     msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

#     update_label(msg)

def test_model():
    global fn
    if fn!="":
        update_label("Model Testing Start...............")
        
        start = time.time()
    
        X=test_model_proc(fn)
        
        X1="Selected Image is {0}".format(X)
        
        end = time.time()
            
        ET="Execution Time: {0:.4} seconds \n".format(end-start)
        
        msg="Image Testing Completed.."+'\n'+ X1 + '\n'+ ET
        fn=""
    else:
        msg="Please Select Image For Prediction...."
        
    update_label(msg)
    
    
def openimage():
   
    global fn
    fileName = askopenfilename(initialdir='E:/Dibetes Mellitus', title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])


#
#        gs = cv2.cvtColor(cv2.imread(imgpath, 1), cv2.COLOR_RGB2GRAY)
#
#        gs = cv2.resize(gs, (x1, y1))
#
#        retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root, image=imgtk, height=250, width=250)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250)
    #result_label1.place(x=300, y=100)
    img.image = imgtk
    img.place(x=300, y=100)
   # out_label.config(text=imgpath)

def convert_grey():
    global fn    
    IMAGE_SIZE=200
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250, font=("bold", 25), bg='bisque2', fg='black',height=250)
    #result_label1.place(x=300, y=400)
    img2 = tk.Label(root, image=imgtk, height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250, font=("bold", 25), bg='bisque2', fg='black')
    #result_label1.place(x=300, y=400)


#################################################################################################################
def window():
    root.destroy()
from tkinter import messagebox as ms


button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="maroon",fg="white")
button1.place(x=30, y=100)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="maroon",fg="white")
button2.place(x=30, y=200)

# # 
button4 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model,width=15, height=1,bg="maroon",fg="white", font=('times', 15, ' bold '))
button4.place(x=30, y=300)


button3 = tk.Button(frame_alpr, text="Train Model", command=train_model, width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button3.place(x=30, y=400)

# button5 = tk.Button(root, text="ULCER", command=window,width=8, height=1, font=('times', 15, ' bold '),bg="yellow4",fg="white")
# button5.place(x=450, y=250)

# button5 = tk.Button(root, text="DIBETES MELLITUS", command=window,width=18, height=1, font=('times', 15, ' bold '),bg="yellow4",fg="white")
# button5.place(x=750, y=350)

#
#button5 = tk.Button(frame_alpr, text="button5", command=window,width=8, height=1, font=('times', 15, ' bold '),bg="yellow4",fg="white")
#button5.place(x=450, y=20)


exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1,bg="red", font=('times', 15, ' bold '),fg="white")
exit.place(x=30, y=500)



root.mainloop()