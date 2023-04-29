import os
import pandas as pd
from misc import is_img
from get_voucher_coords import ImageToCoords
import keras_ocr
import matplotlib.pyplot as plt
from threading import Thread
from time import sleep
from tkinter import *
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Users/daniel.woodrich/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

#voucher dimensions (10px):
#this will be used to reference relative coordinates of different fields
#VOUCHER_DIMS = [[0,0],[118,0],[0,95],[118,95]]
VOUCHER_DIMS = [[0,0],[442,0],[0,352],[442,352]]

#make this more customizable later
IMAGEDIR = "Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3"
#IMAGEDIR = "Y:/RACE_Imagery/Field_Photos/EBSshelf2021/Vesteraalen/Leg3/EBS"
IMAGEFILES = os.listdir(IMAGEDIR)

#IMAGEDIR = "C:/Users/daniel.woodrich/Pictures"
#IMAGEFILES = os.listdir(IMAGEDIR)

IMAGEFILES = [IMAGEDIR + "/" + f for f in IMAGEFILES if is_img(f)]

#one static dictionary that will store the label field coordinates

#used crop of [Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3/EBS_94_202201_L3_0046.JPG] as template image
#start this with a few values to test pipeline, finish it out later.
#[tl,tr,bl,br]

#going to seperate this into a labels and fields dict. labels dict will only contain unique words, since these
#will be used to reposition voucher.

#"title_specimen":[[24.5,7.5],[47.5,7.5],[24.5,11.5],[47.5,11.5]],
VOUCHER_DIMS_DICT = {"labels": {
                        "collection":[[181.,  25.],
       [289.,  25.],
       [289.,  41.],
       [181.,  41.]],
                        "label":[[290.76852 ,  24.582926],
       [347.91806 ,  23.312931],
       [348.26553 ,  38.949654],
       [291.116   ,  40.219646]],

                        "center": [[375.,  88.],
       [415.,  88.],
       [415., 100.],
       [375., 100.]],
                        #"sand": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"point": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"way": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "vessel": [[ 15.06996 , 134.20174 ],
       [ 65.26358 , 135.45659 ],
       [ 64.920044, 149.19801 ],
       [ 14.726423, 147.94316 ]],
                        "cruise": [[153., 125.],
       [201., 125.],
       [201., 137.],
       [153., 137.]],
                        #"haul": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "stomach": [[ 15.042906, 182.36955 ],
       [ 78.288574, 184.03392 ],
       [ 77.9557  , 196.68306 ],
       [ 14.710035, 195.01869 ]],
                        "right": [[224., 185.],
       [264., 185.],
       [264., 196.],
       [224., 196.]],
                        "left": [[323., 187.],
       [355., 187.],
       [355., 198.],
       [323., 198.]],
                        "whole": [[ 14., 217.],
       [ 60., 217.],
       [ 60., 228.],
       [ 14., 228.]],
                        "animal": [[ 13., 228.],
       [ 61., 228.],
       [ 61., 240.],
       [ 13., 240.]],
                        "length": [[133., 230.],
       [186., 230.],
       [186., 243.],
       [133., 243.]],
                        "weight": [[282., 232.],
       [333., 232.],
       [333., 245.],
       [282., 245.]],
                        "identification": [[ 69., 256.],
       [168., 256.],
       [168., 269.],
       [ 69., 269.]],
                        "comments": [[ 13., 279.],
       [ 86., 279.],
       [ 86., 292.],
       [ 13., 292.]],
                        #"collector's": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "initials": [[100., 328.],
       [153., 328.],
       [153., 341.],
       [100., 341.]],
                        "preservative": [[246.05098, 329.26624],
       [340.3224 , 332.03894],
       [339.91272, 345.96808],
       [245.64131, 343.19537]]
},
                     "fields": {
                        "vessel":[[18.5*3.745,34*3.705],[40*3.745,34*3.705],[18.5*3.745,40*3.705],[40*3.745,40*3.705]],
                        "cruise_number": [[55.5*3.745, 34*3.705], [78*3.745, 34*3.705], [55.5*3.745, 40*3.705], [78*3.745, 40*3.705]],
                        "haul_number": [[94*3.745, 34*3.705], [113*3.745, 34*3.705], [94*3.745, 40*3.705], [113*3.745, 40*3.705]],
                        "specimen_number": [[37*3.745, 41.5*3.705], [113*3.745, 41.5*3.705], [37*3.745, 47.5*3.705], [113*3.745,47.5*3.705]],
                        #"stomach_sample": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"tissue_sample": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"right_ovary": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"left_ovary": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"whole_animal": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"length_cm": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"weight_gm": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "species_identification": [[46.5*3.745, 67*3.705], [113*3.745, 67*3.705], [46.5*3.745, 72*3.705], [113*3.745, 72*3.705]],
                        #"comments": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"colectors_initials": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "preservative": [[94*3.745, 86*3.705], [113*3.745, 86*3.705], [94*3.745, 92*3.705], [113*3.745, 92*3.705]]
}}

#data will be stored in a table- read this from file at script start.
DATA = pd.read_csv("data.csv")

class Program:

    #can make this more specific to describe images needing processed as well as analyzed.
    def __init__(self,images_in,data_in):
        #once have some data, do
        if data_in.empty:
            self.image_to_go = images_in
        else:
            self.image_to_go = [f for f in images_in if f not in data_in.images_fullpath]
        self.processed_image_coords = {} #key is image name, each is a dict of 1 coords and 2 path to image crop.
        self.reviewed_image_data = {} #

    def process_images(self):
        self.kocr = keras_ocr.pipeline.Pipeline()
        for n in range(len(self.image_to_go)):
            self.process_image(n)

    def process_image(self,i):

        #run the pipeline on a single image. output data into self.data
        focal_img = self.image_to_go[i]

        print(focal_img)

        #populates data needed for review method
        imagetocoords= ImageToCoords(self.kocr,focal_img,VOUCHER_DIMS,VOUCHER_DIMS_DICT)

        #do it
        imagetocoords.get_coords()

        #after this runs, register image as having been completed.

        #from this initiate two proceses. 1- loop through fields and send to hugging face algo to
        #predict data present. save these in data, with reviewed column set to 0

        #also, have different threaded process running to present data ready (reviewed column 0) to review to user
    #kind of annoying but will have to independently assign all tkinter label methods in this style.
    def test(self):
        self.lbl.configure(text="I just got clicked")
    def review_data(self):
        # create root window
        root = Tk()

        # root window title and dimension
        root.title("Do nothing")
        # Set geometry (widthxheight)
        root.geometry('350x200')

        self.lbl = Label(root, text="haven't been clicked")
        self.lbl.grid()

        btn = Button(root, text="Click me",
                     fg="red", command= self.test)

        btn.grid(column=1, row=0)

        root.mainloop()

    #if I run into performance issues, I can try feeding chunks of images to pipeline, perhaps asynchronously
    def run(self):
        #run these methods on different threads

        #thread 1
        thread1 = Thread(target=self.process_images).start()

        #thread 2:
        #thread2 = self.save_sometimes()

        #thread 3: let user visualize data
        #wait for data to appear before loading up gui
        #while(len(self.data==0)):
        #    sleep(5)

        self.review_data()


#complete version of this should allow user to either process data, review data, or do both.

Program(IMAGEFILES,DATA).run()


#                        "collection":[[48.5,7.5],[77,7.5],[48.5,11.5],[77,11.5]],
#                        "label":[[78,7.5],[92.5,7.5],[78,11.5],[92.5,11.5]],
#                        "national":[[4.7,25.5],[17,25.5],[4.7,28.5],[17,28.5]],
#                        "service":[[42,25.5],[53.5,25.5],[42,28.5],[53.5,28.5]],
#                        "marine":[[17.5,25.5],[27.5,25.5],[17.5,28.5],[27.5,28.5]],
#                        "alaska": [[62.5, 25.5], [73, 25.5], [62.5, 28.5], [73, 28.5]],
#                        "science": [[87.5, 25.5], [100, 25.5], [87.5, 28.5], [100, 28.5]],
#                        "center": [[100.5, 25.5], [110.5, 25.5], [100.5, 28.5], [110.5, 28.5]],
                        #"sand": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"point": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"way": [[0, 0], [0, 0], [0, 0], [0, 0]],
#                        "vessel": [[4.3,38],[17.5,38],[4.3,40.5],[17.5,40.5]]


#"vessel": [[18.5, 34], [40, 34], [18.5, 40], [40, 40]],
#"cruise_number": [[55.5, 34], [78, 34], [55.5, 40], [78, 40]],
#"haul_number": [[94, 34], [113, 34], [94, 40], [113, 40]],
#"specimen_number": [[37, 41.5], [113, 41.5], [37, 47.5], [113, 47.5]],
# "stomach_sample": [[0, 0], [0, 0], [0, 0], [0, 0]],
# "tissue_sample": [[0, 0], [0, 0], [0, 0], [0, 0]],
# "right_ovary": [[0, 0], [0, 0], [0, 0], [0, 0]],
# "left_ovary": [[0, 0], [0, 0], [0, 0], [0, 0]],
# "whole_animal": [[0, 0], [0, 0], [0, 0], [0, 0]],
# "length_cm": [[0, 0], [0, 0], [0, 0], [0, 0]],
# "weight_gm": [[0, 0], [0, 0], [0, 0], [0, 0]],
#"species_identification": [[46.5, 67], [113, 67], [46.5, 72], [113, 72]],
# "comments": [[0, 0], [0, 0], [0, 0], [0, 0]],
# "colectors_initials": [[0, 0], [0, 0], [0, 0], [0, 0]],
#"preservative": [[94, 86], [113, 86], [94, 92], [113, 92]]