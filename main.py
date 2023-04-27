import os
import pandas as pd
from misc import is_img
from get_voucher_coords import ImageToCoords
import keras_ocr
import matplotlib.pyplot as plt
from threading import Thread
from time import sleep
from tkinter import *

#voucher dimensions (10px):
#this will be used to reference relative coordinates of different fields
VOUCHER_DIMS = [[0,0],[118,0],[0,95],[118,95]]

#make this more customizable later
IMAGEDIR = "Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3"
IMAGEFILES = os.listdir(IMAGEDIR)

IMAGEFILES = [IMAGEDIR + "/" + f for f in IMAGEFILES if is_img(f)]

#one static dictionary that will store the label field coordinates

#used crop of [Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3/EBS_94_202201_L3_0046.JPG] as template image
#start this with a few values to test pipeline, finish it out later.
#[tl,tr,bl,br]

#going to seperate this into a labels and fields dict. labels dict will only contain unique words, since these
#will be used to reposition voucher.

#"title_specimen":[[24.5,7.5],[47.5,7.5],[24.5,11.5],[47.5,11.5]],
VOUCHER_DIMS_DICT = {"labels": {
                        "collection":[[48.5,7.5],[77,7.5],[48.5,11.5],[77,11.5]],
                        "label":[[78,7.5],[92.5,7.5],[78,11.5],[92.5,11.5]]
                        #"national":[[0,0],[0,0],[0,0],[0,0]],
                        #"service":[[0,0],[0,0],[0,0],[0,0]],
                        #"marine":[[0,0],[0,0],[0,0],[0,0]],
                        #"alaska": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"science": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"center": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"sand": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"point": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"way": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"vessel": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"cruise": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"haul": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"stomach": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"right": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"left": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"whole": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"animal": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"length": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"weight": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"identification": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"comments": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"collector's": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"initials": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"preservative": [[0, 0], [0, 0], [0, 0], [0, 0]],
},
                     "fields": {
                        "vessel":[[18.5,34],[40,34],[18.5,40],[40,40]],
                        #"cruise_number": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"haul_number": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"specimen_number": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"stomach_sample": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"tissue_sample": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"right_ovary": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"left_ovary": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"whole_animal": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"length_cm": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"weight_gm": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "species_identification": [[46.5, 67], [113, 67], [46.5, 72], [113, 72]]
                        #"comments": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"colectors_initials": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"preservative": [[0, 0], [0, 0], [0, 0], [0, 0]]
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