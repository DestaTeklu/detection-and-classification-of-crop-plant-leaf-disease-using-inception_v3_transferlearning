

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




from tkinter import *
from tkinter import ttk
from PIL import Image
from tkinter import filedialog
from tkinter import messagebox
import cv2
from urllib.request import urlopen
import matplotlib.pyplot as plt
#detection
from PIL import Image, ImageTk
def Import_Image():
    #dir = (os.path.dirname(filename)
    #if filename is not None else "."
    
    filename = filedialog.askopenfilename(title="Bookmarks - Open File",filetypes=[("Bookmarks files","*.jpg")],defaultextension=".jpg")
    imagename.set(filename)
    #image= filedialog.askopenfilename()initialdir=dir,
    #imagename=image
    #print(filename)
   
    image=cv2.imread(filename)
    cv2.imshow("original image",image)
    #image=detector_method(filename)
    #cv2.imshow('imported image',image)
    img = ImageTk.PhotoImage(file=filename)
    #treeview1.insert('','end',text='leaf',image=img)
    #text.image_create('insert',image=img)
    #cv2.imshow("image",image)
    #in_frame = ttk.Label(labelframe, image = img)
    #in_frame.pack()
    #canvas.create_image(20,20, anchor=NW, image=img)



import numpy as np
#contour
def detect():
    filename=imagename.get()
    print(filename)
    if filename=="":
        messagebox.showwarning(title='error',message="please upload Image first !")
        print("please select a photo first")
    else:
        img=cv2.imread(filename)
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.title('original image')
        plt.xticks([])
        plt.yticks([])
        img1=img
        '''ret, thresh_img=cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),175,255,cv2.THRESH_BINARY)
        image,contour,hier=cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for a in contour:
            (x,y),radius=cv2.minEnclosingCircle(a)
            center=(int(x),int(y))
            radius=int(radius)
            img=cv2.circle(img,center,radius,(0,255,255),2)'''
        
        ret, thresh_img= cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 40,255,cv2.THRESH_BINARY_INV)
        image,contour,hier=cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        
        for c in contour:
            '''
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
            rect=cv2.minAreaRect(c)
            box= cv2.boxPoints(rect)
            box=np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255))'''
            (x,y),radius=cv2.minEnclosingCircle(c)
            center=(int(x),int(y))
            radius=int(radius)
            img=cv2.circle(img,center,radius,(0,255,255),2)
            #cv2.drawContours(img,contour,-1,(0,0,255),1)
        #bgs = cv2.BackgroundSubtractorMOG2()
        #cv2.namedWindow("Original", 1)
        #cv2.namedWindow("Foreground", 1)
        #cv2.imshow("Original", img1)
        #fgmask = bgs.apply(img1)
        #foreground = cv2.bitwise_and(img1, img1, mask=fgmask)
        #cv2.imshow("Foreground", foreground)
        #cv2.imshow("Original Image",img1)

        plt.subplot(2,2,2)
        plt.imshow(thresh_img)
        plt.title('thresholded Image')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(2,2,3)
        plt.imshow(img)
        plt.title('Detected Image')
        plt.xticks([])
        plt.yticks([])

        plt.show()
        #cv2.imshow('Detected Image', img)
        
        
        #cv2.imshow("thresholded Image", thresh_img)

        #cv2.waitKey(0)
        #& 0XFF == ord('q')
        #if k==27:
         #   break
        #cv2.destroyAllWindows()
        


#classification


import argparse

import numpy as np
import tensorflow as tf


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def clasify():
    filename= imagename.get()
    if filename=="":
        messagebox.showwarning(title='error',message="please select an image first")
        
    else:
        
          file_name = filename
          image=cv2.imread(file_name)
          #cv2.imshow("classified image",image)
          model_file ="output_graph.pb"
          label_file = "labels.txt"
          input_height = 299
          input_width = 299
          input_mean = 0
          input_std = 255
          input_layer = "Mul"
          output_layer = "final_result"

          parser = argparse.ArgumentParser()
          parser.add_argument("--image", help="image to be processed")
          parser.add_argument("--graph", help="graph/model to be executed")
          parser.add_argument("--labels", help="name of file containing labels")
          parser.add_argument("--input_height", type=int, help="input height")
          parser.add_argument("--input_width", type=int, help="input width")
          parser.add_argument("--input_mean", type=int, help="input mean")
          parser.add_argument("--input_std", type=int, help="input std")
          parser.add_argument("--input_layer", help="name of input layer")
          parser.add_argument("--output_layer", help="name of output layer")
          args = parser.parse_args()

          if args.graph:
            model_file = args.graph
          if args.image:
            file_name = args.image
          if args.labels:
            label_file = args.labels
          if args.input_height:
            input_height = args.input_height
          if args.input_width:
            input_width = args.input_width
          if args.input_mean:
            input_mean = args.input_mean
          if args.input_std:
            input_std = args.input_std
          if args.input_layer:
            input_layer = args.input_layer
          if args.output_layer:
            output_layer = args.output_layer

          graph = load_graph(model_file)
          t = read_tensor_from_image_file(
              file_name,
              input_height=input_height,
              input_width=input_width,
              input_mean=input_mean,
              input_std=input_std)

          input_name = "import/" + input_layer
          output_name = "import/" + output_layer
          input_operation = graph.get_operation_by_name(input_name)
          output_operation = graph.get_operation_by_name(output_name)

          with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
          results = np.squeeze(results)

          top_k = results.argsort()[-5:][::-1]
          labels = load_labels(label_file)
          for i in top_k:
            print(labels[i], results[i]*100)
            #label+i=ttk.Label(labelframe,textvariable=labels[i], results[i]).pack()
            #textinput.insert("1.0 + 2 lines lineend",labels[i], results[i]*100 + '\n')
            treeview.insert('','end',text=labels[i]+'=='+str(results[i]*100) )
            
            





        
if __name__ == '__main__':    
    root = Tk()
    #root.fileName=image=filedialog.askopenfilename(filetypes=(("All files","*.jpg")))

    #root.geometry("850x500")
    root.title("Plant Leaf Disease Detection and Classification")
    
    pannedwindow=ttk.Panedwindow(root,orient=VERTICAL)
    pannedwindow.pack(fill=BOTH,expand=True)

    style=ttk.Style()

    #frame.pack()
    #frame.config(height=400,width=400)
    #frame.config(relief=RIDGE)
    imagename=StringVar()
    imagename.set("")
    frame=ttk.Frame(pannedwindow, width=900, height=300, relief= SUNKEN)
    #frame1= ttk.Frame(pannedwindow,width=200,height=0, relief=SUNKEN)
    #frame3= ttk.Frame(pannedwindow,width=200,height=200, relief=SUNKEN)
    frame2= ttk.Frame(pannedwindow,width=400,height=300, relief=SUNKEN)
    #label=ttk.Label(frame1,textvariable=imagename)

    title=ttk.Label(frame, text="PLANT LEAVE DISEASE IDENTIFIER", font=('Arial',24),foreground='green')
    #title.grid(row=20,column=1,columnspan=90,rowspan=3,padx=5,pady=5,ipadx=5,ipady=5)
    title.place(x=100,y=1,height=150)
    Upload_Image= ttk.Button(frame,text="Import Image")
    #Upload_Image.pack(side=LEFT, padx=10, pady=10 )
    Upload_Image.config(command=Import_Image)
    Upload_Image.place(x=100,y=200,height=40)

    detect=ttk.Button(frame,text="Detect", command= detect)
    #detect.grid(row=104, column=100,columnspan=3,rowspan=3,padx=10,pady=10,ipadx=5,ipady=5)
    detect.place(x=250,y=200,height=40)

    classify=ttk.Button(frame,text="Classify", command= clasify)
    #classify.grid(row=104,column=104,columnspan=3,rowspan=3,padx=5,pady=5,ipadx=5,ipady=5)
    classify.place(x=400,y=200,height=40)
    frame.config(padding=(50,50))

    style.configure('TButton',foreground='blue',background='green',font=('Arial',10,'bold'))
    #Upload_Image.config('Alarm.TButton')
    
    #detect.pack(side=RIGHT, padx=5, pady=5

    #label frame
    #labelframe=ttk.LabelFrame(frame3, height=200, width=200, text='Display result').pack(fill=BOTH,expand=True)
    #text=Text(frame3,width=20,height=10)
    #text.pack(fill=BOTH,expand=True)
    #text
    treeview=ttk.Treeview(frame2)
    treeview.pack(fill=BOTH,expand=True)
    #image
    

    #image=PhotoImage(file=filename)
    #textinput.image_create('insert',image=image)
    #canvas = Canvas(root, width = 50, height = 50)      
    #canvas.pack(fill=BOTH, expand=True) 

    pannedwindow.add(frame, weight=3)
    #pannedwindow.add(frame1,weight=1)
    #pannedwindow.add(frame3, weight=3)
    pannedwindow.add(frame2, weight=3)


    '''label=ttk.Label(root,textvariable=" ")
    label.grid(row=3,column=1)

    #classify.pack(side=RIGHT)'''
    root.mainloop()
