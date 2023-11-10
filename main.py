import os
import pandas as pd
#import numpy as np
from misc import is_img, CreateToolTip
#import matplotlib.pyplot as plt
#from threading import Thread
#from time import sleep
import tkinter as tk
from copy import deepcopy
from PIL import ImageTk, Image
import code
import pathlib

#---
#libraries for img processing (turn on when using)

from image_process import ImageToData
import keras_ocr
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

pytesseract.pytesseract.tesseract_cmd = 'C:/Users/daniel.woodrich/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

#parameter to turn on and off label prediction
PREDICT_FIELDS = True

#---

#paramters to change:

mode = 'process' #review or process
#DATAPATH = "data_base.csv" #this was processed from the source path: "Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3" . using base ocr model
#DATAPATH = "data_largetrocr.csv"
DATAPATH = "data_new.csv"
#training data is saved in
#DATAPATH = "data_all.csv" #this applies to all imagery, but starts on a cruise from the 2000s.

#make this more customizable later
#IMAGEDIRS = ["Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3","Y:/RACE_Imagery/Field_Photos/AI2018Photos","Y:/RACE_Imagery/Field_Photos/EBSshelf2004"]
IMAGEDIRS = ["Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3"]

##################################


#when changing data source, make sure to rename training data correctly (for now).

import cv2

#load trocr model

#image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

#pixel_values = processor(images=image, return_tensors="pt").pixel_values

#tmp off



#IMAGEFILES = os.listdir(IMAGEDIR)
#IMAGEFILES = [IMAGEDIR + "/" + f for f in IMAGEFILES if is_img(f)]
IMAGEFILES = []
for i in IMAGEDIRS:
    IMAGEDIR = pathlib.Path(i)
    add_files=list(IMAGEDIR.rglob("*.[jJ][pP][gG]")) #[png][gif][jpg][JPG][jpeg]

    IMAGEFILES = IMAGEFILES + [str(f) for f in add_files]


#make the IMAGEFILES correspond to all of the

#one static dictionary that will store the label field coordinates

#used crop of [Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3/EBS_94_202201_L3_0046.JPG] as template image
#start this with a few values to test pipeline, finish it out later.
#[tl,tr,bl,br]

#going to seperate this into a labels and fields dict. labels dict will only contain unique words, since these
#will be used to reposition spec_col_label.

#"title_specimen":[[24.5,7.5],[47.5,7.5],[24.5,11.5],[47.5,11.5]],
#spec_col_label dimensions (10px):
#this will be used to reference relative coordinates of different fields
#spec_col_label_DIMS = [[0,0],[118,0],[0,95],[118,95]]
SPEC_COL_LABEL_DIMS = [[0,0],[442,0],[0,352],[442,352]]
SPEC_COL_LABEL_DIMS_DICT = {"labels": {
                        "collection":[[181.,  25.],[289.,  25.],[289.,  41.],[181.,  41.]],
                        "label":[[290.76852 ,  24.582926],[347.91806 ,  23.312931],[348.26553 ,  38.949654],[291.116   ,  40.219646]],
                        "center": [[375.,  88.],[415.,  88.],[415., 100.],[375., 100.]],
                        #"sand": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"point": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        #"way": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "vessel": [[ 15.06996 , 134.20174 ],[ 65.26358 , 135.45659 ],[ 64.920044, 149.19801 ],[ 14.726423, 147.94316 ]],
                        "cruise": [[153., 125.],[201., 125.],[201., 137.],[153., 137.]],
                        #"haul": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "stomach": [[ 15.042906, 182.36955 ],[ 78.288574, 184.03392 ],[ 77.9557  , 196.68306 ],[ 14.710035, 195.01869 ]],
                        "right": [[224., 185.],[264., 185.],[264., 196.],[224., 196.]],
                        "left": [[323., 187.],[355., 187.],[355., 198.],[323., 198.]],
                        "whole": [[ 14., 217.],[ 60., 217.],[ 60., 228.],[ 14., 228.]],
                        "animal": [[ 13., 228.],[ 61., 228.],[ 61., 240.],[ 13., 240.]],
                        "length": [[133., 230.],[186., 230.],[186., 243.],[133., 243.]],
                        "weight": [[282., 232.],[333., 232.],[333., 245.],[282., 245.]],
                        "identification": [[ 69., 256.],[168., 256.],[168., 269.],[ 69., 269.]],
                        "comments": [[ 13., 279.],[ 86., 279.],[ 86., 292.],[ 13., 292.]],
                        #"collector's": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "initials": [[100., 328.],[153., 328.],[153., 341.],[100., 341.]],
                        "preservative": [[246.05098, 329.26624],[340.3224 , 332.03894],[339.91272, 345.96808],[245.64131, 343.19537]]
},
                     "fields": {
                        "vessel":[[69.2825,118],[149.8,118],[69.2825,148.2],[149.8,148.2]],
                        "cruise_number": [[207.8475, 118], [292.11, 118], [207.8475, 148.2], [292.11, 148.2]],
                        "haul_number": [[352, 118], [433.185, 118], [352, 148.2], [433.185, 148.2]],
                        "specimen_number": [[138.565, 150.7575], [423.185, 150.7575], [138.565, 175.9875], [423.185, 175.9875]],
                        "stomach_sample": [[75, 185],[120, 185],[75, 210],[ 120, 210]],
                        "tissue_sample": [[170, 185],[225, 185],[170, 210],[225 , 210]],
                        "right_ovary": [[270, 185], [325, 185], [270, 210], [325, 210]],
                        "left_ovary": [[365, 185], [433, 185], [365, 210], [433, 210]],
                        "whole_animal": [[69.2825, 215], [120, 215], [69.2825, 243], [120, 243]],
                        "length_cm": [[215, 215], [285, 215], [215, 243], [285, 243]],
                        "weight_gm": [[365, 215], [433, 215], [365, 243], [433, 243]],
                        "species_identification": [[174.1425, 242.235], [433.185, 242.235], [174.1425, 270.76], [433.185, 270.76]],
                        "comments": [[15, 273], [425, 273], [15, 318], [425, 318]],
                        "collector_initials": [[155, 318.63], [243, 318.63], [155, 350], [243, 350]],
                        "preservative": [[347, 318.63], [433.185, 318.63], [347, 350], [433.185, 350]]
}}

FIELD_NAMES_DATA = [a for a in SPEC_COL_LABEL_DIMS_DICT["fields"]]
#data keys should match output csv columns
DATA_KEYS = {a:None for a in ["id","spec_col_label_id","image_fullpath","image_croppath"]+["pred_"+ b for b in FIELD_NAMES_DATA]+["verified_" + c for c in FIELD_NAMES_DATA]}

FIELD_NAMES_APP = ["VESSEL","CRUISE NUMBER","HAUL NUMBER","SPECIMEN NUMBER","STOMACH SAMPLE",
                   "TISSUE SAMPLE","RIGHT OVARY","LEFT OVARY","WHOLE ANIMAL","LENGTH (CM)",
                   "WEIGHT (GM)","SPECIES IDENTIFICATION","COMMENTS","COLLECTOR'S INITIALS",
                   "PRESERVATIVE"]


class FormField():

    def __init__(self,frame,labtext,datalab,labnum):
        #super().__init__(*args, **kwargs)
        self.label = tk.Label(frame, text=labtext).grid(row=labnum, column=0, sticky="w")
        self.entry = tk.Entry(frame)
        self.entry.grid(row=labnum, column=1, padx=5)
        self.pred_labname = "pred_" + datalab
        self.verified_labname = "verified_" + datalab

        self.dissallowed_prefixes = [".",". "," .","#"," #","# "]

        #specific behavior for comments, which includes the word comment in the two line field
        if datalab =='comments':
            self.hardcode_disallowed = ['comments','comments',"conserro.","conserro","commercials.","commercials","convermo.","convermo","comments # #","comments #",
                                        "commercialists","commercialists.","contents #","contents","contents."]
        else:
            self.hardcode_disallowed = []

    #compare to pred_correct_dict and dissalowed keys here:
    def refresh_entry(self,cur_data,pop_predictions,pred_correct_dict,disallowed_keys):
        #print(cur_data[self.pred_labname])
        self.entry.delete(0, tk.END)
        #if cur_data[self.verified_labname].isna().bool():
        if (cur_data[self.verified_labname]=="").bool():
            #if not cur_data[self.pred_labname].isna().bool() and pop_predictions:
            if not (cur_data[self.pred_labname]=="").bool() and pop_predictions:
                if cur_data.loc[0,self.pred_labname] in pred_correct_dict and cur_data.loc[0,self.pred_labname] not in disallowed_keys:
                    pred = pred_correct_dict[cur_data.loc[0,self.pred_labname]]
                else:
                    pred = cur_data.loc[0,self.pred_labname]

                for l in self.hardcode_disallowed:
                    pred = pred.replace(l,'')

                #add this in once I update python past 3.11
                #for m in self.dissallowed_prefixes:
                #    pred = pred.removeprefix(m)
                for m in self.dissallowed_prefixes:
                    lm = len(m)
                    if pred[:lm]==m:
                        pred = pred[lm:]

                #remove all whitespace from start and end.
                pred = pred.strip()

                self.entry.insert(0, pred)
            else:
                self.entry.insert(0, "")
        else:
            self.entry.insert(0, cur_data.loc[0,  self.verified_labname])

class Program:

    #can make this more specific to describe images needing processed as well as analyzed.
    def __init__(self,images_in,data_in,datakeys,field_names_app,field_names_data):
        #once have some data, do
        self.datakeys = datakeys
        self.datapath = data_in
        self.field_names_app = field_names_app
        self.field_names_data = field_names_data
        #code.interact(local=locals())
        self.data = pd.read_csv(data_in, dtype = str,keep_default_na=False)
        if self.data.empty:
            self.image_to_go = images_in
        else:
            self.image_to_go = [f for f in images_in if f not in self.data.image_fullpath.to_list()]
        self.processed_image_coords = {} #key is image name, each is a dict of 1 coords and 2 path to image crop.
        self.reviewed_image_data = {} #
        self.prev_array ={}
        #populate below by looking in loaded data
        #code.interact(local=locals())
        #code.interact(local=locals())
        self.max_spec_col_label_id = pd.to_numeric(self.data.spec_col_label_id).max()
        self.max_row_id = pd.to_numeric(self.data.id).max()

        #test for NA, give 0 if na
        if self.max_spec_col_label_id != self.max_spec_col_label_id:
            #code.interact(local=locals())
            self.max_spec_col_label_id = 1

        self.max_spec_col_label_id = int(self.max_spec_col_label_id)

        # test for NA, give 0 if na
        if self.max_row_id != self.max_row_id:
            self.max_row_id = 1
        else:
            self.max_row_id += 1

        self.max_row_id = int(self.max_row_id)

        #create a hashset of dissalowed keys
        self.disallowed_keys = set()

        #create a dict of predicted and verified responses
        self.pred_correct_dict = {}

        for i in range(len(self.data)):
            row = self.data.iloc[[i]].reset_index(drop=True)
            self.populate_pred_lookup(row)

        #populate a list of rows where spec_col_label is populated
        self.spec_col_idxs = [i for i, x in enumerate(self.data.spec_col_label_id) if x != ""]

        #print(self.pred_correct_dict)
        #print(self.disallowed_keys)

    def populate_pred_lookup(self,row):

        #print(row)
        #print(row.dtypes)

        for m in self.field_names_data:
            v_ans=row.loc[0,"verified_" + m]
            p_ans=row.loc[0,"pred_" + m]

            #v_ans = row["verified_" + m].values
            #p_ans = row["pred_" + m].values
            #print(type(v_ans))
            # only assess if row has verified answers, and is not in dissalowed keys
            if not v_ans=="" and p_ans not in self.disallowed_keys:
                if p_ans != v_ans:
                    # add to pred correct dict
                    self.pred_correct_dict[p_ans] = v_ans
                else:
                    # if the prediction was ever correct, allow it to show up again
                    self.disallowed_keys.add(p_ans)

        #print(self.pred_correct_dict)
        #print(self.disallowed_keys)

    def process_images(self):
        self.kocr = keras_ocr.pipeline.Pipeline()
        #self.tr_ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        #elf.trocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        self.tr_ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.trocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

        for n in range(len(self.image_to_go)):
            self.process_image(n)

    def process_image(self,i):

        #run the pipeline on a single image. output data into self.data
        focal_img = self.image_to_go[i]

        print(focal_img)

        #populates data needed for review method
        print(self.max_spec_col_label_id)

        datain = deepcopy(self.datakeys)
        datain["id"] = self.max_row_id
        datain["image_fullpath"] = focal_img

        imagetodata= ImageToData(self.kocr,self.tr_ocr_processor,self.trocr,focal_img,SPEC_COL_LABEL_DIMS,SPEC_COL_LABEL_DIMS_DICT,self.prev_array,self.max_spec_col_label_id,datain,PREDICT_FIELDS)


        #do it
        data,prev_array = imagetodata.get_data()

        if not data["spec_col_label_id"] == None:
            self.max_spec_col_label_id = data["spec_col_label_id"]
            #code.interact(local=locals())

        self.prev_array = prev_array

        self.max_row_id = self.max_row_id + 1

        row = pd.DataFrame(data, index=[0, ])

        row.to_csv(self.datapath, mode='a',index=False,header=False)

        #save data to csv. if max_spec_col_label_id is none, change it to NA (no spec_col_label present). still good to document
        #for future discovery.

    def review_data(self):

        #use verified vessel as a proxy
        #self.cur_ind = self.data.verified_vessel.isna().idxmin()-1

        #self.cur_ind = (self.data.verified_vessel=="").idxmin() - 1
        #print((self.data.verified_vessel==""))
        ser = (self.data.verified_vessel!="")
        self.cur_ind = ser.where(ser).last_valid_index()
        #print(ser)
        #print(self.cur_ind)
        #print(self.spec_col_idxs)
        if self.cur_ind != None:
            self.spec_col_label_img_ind = self.spec_col_idxs.index(self.cur_ind+1)
        else:
            self.spec_col_label_img_ind =0
            self.cur_ind =0
        #print(self.cur_ind)

        #self.cur_encountered = []

        # code.interact(local=locals())
        #self.image_advance()

        self.lock_spec_col_label = False
        self.pop_predictions = True
        self.advance_on_save = True

        # create root window
        window = tk.Tk()
        window.title("Field Image Annotation App")

        # Create frames for the left column, top-right quadrant, and bottom-right quadrant
        left_frame = tk.Frame(window)
        top_right_frame = tk.Frame(window)

        # Configure grid layout
        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        left_frame.grid(row=0, column=0, sticky="nsew")
        top_right_frame.grid(row=0, column=1, sticky="nsew")

        cropped_buttons_frame = tk.Frame(top_right_frame)
        cropped_buttons_frame.pack()

        self.full_image_label = tk.Label(left_frame)
        #self.full_image_label.place(anchor = tk.NW)

        self.full_image_label.pack(anchor = tk.NW)

        self.zoom_rotate_button = tk.Button(left_frame, text="zoom/rotate", command=lambda: self.full_image_zoom(window))
        self.zoom_rotate_button.place(anchor = tk.NW)

        self.last_cropped_image_button = tk.Button(cropped_buttons_frame, text="Previous spec_col_label",
                                              command=lambda: self.advance_spec_col_label(-1))
        self.last_cropped_image_button.grid(row=0, column=0)

        CreateToolTip(self.last_cropped_image_button, text='Shortcut: Control-Right key')

        self.refresh_button = tk.Button(cropped_buttons_frame, text="Next spec_col_label", command=lambda: self.advance_spec_col_label(1))
        self.refresh_button.grid(row=0, column=1)

        CreateToolTip(self.refresh_button, text='Shortcut: Control-Left key')

        self.lock_button = tk.Button(cropped_buttons_frame, text="Lock spec_col_label", command=self.toggle_lock)
        self.lock_button.grid(row=1, column=0)

        CreateToolTip(self.lock_button, text='Shortcut: Control-L')

        self.populate_button = tk.Button(cropped_buttons_frame, text="Populate from spec_col_label", command=self.populate_from_spec_col_label)
        self.populate_button.grid(row=1, column=1)

        CreateToolTip(self.populate_button, text='Shortcut: Control-P')

        self.spec_col_label_label = tk.Label(top_right_frame)
        self.spec_col_label_label.pack()

        fields_frame = tk.Frame(top_right_frame)
        fields_frame.pack()

        self.fields = [FormField(fields_frame,self.field_names_app[i],
                                        self.field_names_data[i],i) for i in range(len(self.field_names_data))]

        data_buttons_frame = tk.Frame(top_right_frame)
        data_buttons_frame.pack()

        self.from_last_button = tk.Button(data_buttons_frame, text="Last", command=self.populate_from_last)
        self.from_last_button.grid(row=0, column=0, padx=5)

        CreateToolTip(self.from_last_button, text='Shortcut: Control-R')

        self.pop_predictions_button = tk.Button(data_buttons_frame, text="Predictions", command=self.toggle_pop_predictions)
        self.pop_predictions_button.grid(row=0, column=1, padx=5)
        self.pop_predictions_button.config(relief=tk.SUNKEN)

        CreateToolTip(self.pop_predictions_button, text='Shortcut: Control-O')

        self.clear_data_button = tk.Button(data_buttons_frame, text="Clear", command=self.clear_data)
        self.clear_data_button.grid(row=0, column=2, padx=5)

        CreateToolTip(self.clear_data_button, text='Shortcut: Control-W')

        self.advance_on_save_button = tk.Button(data_buttons_frame, text="Advance on Save",
                                                command=self.toggle_advance_on_save)
        self.advance_on_save_button.grid(row=0, column=3, padx=5)
        self.advance_on_save_button.config(relief=tk.SUNKEN)

        CreateToolTip(self.advance_on_save_button, text='Shortcut: Control-A')

        self.save_button = tk.Button(data_buttons_frame, text="Save", command=self.save_data)
        self.save_button.grid(row=0, column=4, padx=5)

        CreateToolTip(self.save_button, text='Shortcut: Enter key\nShortcut hard save (write to csv): Control-S')

        window.bind("<Down>", lambda e: self.image_cycle(e,-1))
        window.bind("<Up>", lambda e: self.image_cycle(e,1))
        window.bind("<Return>", self.save_data)

        window.bind("<Control-s>",self.hard_save_data)
        window.bind("<Control-r>", self.populate_from_last)
        window.bind("<Control-p>", self.populate_from_spec_col_label)

        window.bind("<Control-Left>", lambda e: self.advance_spec_col_label(e,-1))
        window.bind("<Control-Right>", lambda e: self.advance_spec_col_label(e,1))
        window.bind("<Control-l>", self.toggle_lock)
        window.bind("<Control-o>", self.toggle_pop_predictions)
        window.bind("<Control-w>", self.clear_data)
        window.bind("<Control-a>", self.toggle_advance_on_save)

        self.image_cycle("start",1)

        window.mainloop()

    def rotate_zoom(self,event ='default',direction=-1):

        self.rotate_count += direction

        timg = self.full_image_og.copy()

        timg = timg.resize((self.zwidth,self.zheight)).rotate(90*self.rotate_count,expand=True)

        #self.full_image_og = self.full_image_og.resize((self.zwidth,self.zheight)).rotate(90*direction)

        full_image_tk = ImageTk.PhotoImage(timg)

        #self.zoom_img.create_image(0, 0, image=full_image_tk, anchor="nw")
        self.zoom_img.config(image=full_image_tk)
        self.zoom_img.image = full_image_tk

    def full_image_zoom(self,window):
        zoom_window = tk.Toplevel(window)

        # sets the title of the
        # Toplevel widget
        zoom_window.title("Full image")

        self.rotate_count = 0
        # sets the geometry of toplevel
        #full_image_tk = ImageTk.PhotoImage(Image.open(self.data["image_fullpath"][self.cur_ind]))
        width, height = self.full_image_og.size

        factor = 900/width

        self.zwidth = round(width * factor)
        self.zheight= round(height * factor)

        full_image_tk = ImageTk.PhotoImage(self.full_image_og.resize((self.zwidth,self.zheight)))

        #,width=max(self.zwidth,self.zheight), height=max(self.zwidth,self.zheight)

        left_frame = tk.Frame(zoom_window)

        left_frame.grid(row=0, column=0, sticky="nsew")#, sticky="nsew")

        #self.zoom_img = tk.Canvas(left_frame, height=0, width = 0)
        #self.zoom_img.pack()
        #self.zoom_img.create_image(0,0,image=full_image_tk, anchor="nw")

        self.zoom_img = tk.Label(left_frame, image=full_image_tk)
        self.zoom_img.pack()
        self.zoom_img.image = full_image_tk

        self.rotate_button = tk.Button(left_frame, text="rotate", command = self.rotate_zoom)
        self.rotate_button.place(anchor=tk.NW)

        zoom_window.bind("<Down>", lambda e: self.rotate_zoom(e,1))
        zoom_window.bind("<Left>", lambda e: self.rotate_zoom(e, 1))
        zoom_window.bind("<Up>", lambda e: self.rotate_zoom(e,-1))
        zoom_window.bind("<Right>", lambda e: self.rotate_zoom(e, -1))

        data_string = tk.StringVar()
        data_string.set(self.data["image_fullpath"][self.cur_ind].replace("/","\\"))
        ent = tk.Entry(zoom_window, textvariable=data_string, fg="black", bg="white", bd=0, state="readonly")
        ent.grid(row=1, column=0, sticky="ew")
        #ent.pack(anchor=tk.W)

        #tk.Label(left_frame, text=self.data["image_fullpath"][self.cur_ind]).pack(anchor=tk.W)

    def refresh_full_image(self):
        image_path = self.data["image_fullpath"][self.cur_ind]  # Replace with your image path
        self.full_image_og = Image.open(image_path)
        full_image = self.full_image_og.resize((450, 450))  # Adjust the size as needed
        full_image_tk = ImageTk.PhotoImage(full_image)

        self.full_image_label.config(image=full_image_tk)
        self.full_image_label.image = full_image_tk

    def refresh_data(self,ind):

        self.save_button.config(text="Save")

        self.cur_data = self.data.iloc[[ind]].reset_index(drop=True)

        [field.refresh_entry(self.cur_data,self.pop_predictions,self.pred_correct_dict,self.disallowed_keys) for field in self.fields]

    def image_cycle(self,event='default',direction = 1):

        #populate the full image and the new row of data.
        #populate verified cols that are not na, if na try predicted cols.
        #print(event)

        self.cur_ind = self.cur_ind + direction

        if self.cur_ind < 0 or self.cur_ind > len(self.data)-1:
            self.cur_ind = self.cur_ind - direction
        else:

            self.refresh_full_image()

            #test if image is predicted to be same spec_col_label, if so preload it. only do it in forward direction.
            if self.cur_ind > 0 and direction == 1: #and self.cur_ind not in self.cur_encountered
                #print(self.data.loc[self.cur_ind-1,"spec_col_label_id"])
                #print(self.data.loc[self.cur_ind, "spec_col_label_id"])
                #if self.data.loc[self.cur_ind-1,"spec_col_label_id"] == self.data.loc[self.cur_ind,"spec_col_label_id"] and pd.isna(self.data.loc[self.cur_ind,"verified_vessel"]):
                if self.data.loc[self.cur_ind - 1, "spec_col_label_id"] == self.data.loc[self.cur_ind, "spec_col_label_id"] and not self.data.loc[self.cur_ind, "spec_col_label_id"]=="" and \
                        self.data.loc[self.cur_ind, "verified_vessel"]=="":
                    self.refresh_data(self.cur_ind-1)
                    self.save_button.config(text="Autofilled... Save")
                else:
                    self.refresh_data(self.cur_ind)
            else:
                self.refresh_data(self.cur_ind)

            if not self.lock_spec_col_label:
                #code.interact(local=locals())
                #self.data["spec_col_label_id"].iloc[[self.cur_ind]].equals(self.data["spec_col_label_id"].iloc[[self.cur_ind-1]]) and
                #if not self.data["spec_col_label_id"].iloc[[self.cur_ind]].isna().bool():
                if not (self.data["spec_col_label_id"].iloc[[self.cur_ind]]=="").bool():
                    self.refresh_spec_col_label("automatic",self.cur_ind)

        #self.cur_encountered.append(self.cur_ind)

    def save_data(self,event="default",advance=True):

        self.save_button.config(text="Saved!")

        row = self.data.iloc[[self.cur_ind - 1]].reset_index(drop=True)

        self.populate_pred_lookup(row)

        for field in self.fields:
            ans = field.entry.get()
            if ans == "":
                ans = " " #this will distinguish unanylzed (nothing) from analyzed (space)
            self.data.loc[self.cur_ind, field.verified_labname] = ans

        if advance and self.advance_on_save:
            self.image_cycle(1)

    def hard_save_data(self,event="default"):

        self.save_data(advance=False)

        self.save_button.config(text="Hard Saved!")

        data_recent = pd.read_csv(self.datapath, dtype = str,keep_default_na=False)

        data_out= pd.concat([self.data,data_recent[data_recent['id'].isin(self.data['id']) == False],])

        data_out.to_csv(self.datapath, index=False, header=True)

        if self.advance_on_save:
            self.image_cycle(1)

    def toggle_pop_predictions(self,event="default"):
        self.pop_predictions = not self.pop_predictions

        if self.pop_predictions:
            self.pop_predictions_button.config(relief=tk.SUNKEN)
        else:
            self.pop_predictions_button.config(relief=tk.RAISED)

    def toggle_advance_on_save(self,event="default"):
        self.advance_on_save = not self.advance_on_save

        if self.advance_on_save:
            self.advance_on_save_button.config(relief=tk.SUNKEN)
        else:
            self.advance_on_save_button.config(relief=tk.RAISED)


    def toggle_lock(self,event="default"):
        self.lock_spec_col_label = not self.lock_spec_col_label

        if self.lock_spec_col_label:
            self.lock_button.config(relief=tk.SUNKEN)
        else:
            self.lock_button.config(relief=tk.RAISED)

    def advance_spec_col_label(self,event='default',dir=1):

        #print(self.spec_col_label_img_id+dir)

        #find the last index where spec_col_label id was diff and not na

        if not self.lock_spec_col_label and self.spec_col_label_img_ind < (len(self.spec_col_idxs)-dir) and self.spec_col_label_img_ind  > 0-dir:
            #as of right now, dont use spec_col_label id, just go back on position (that isn't NA)
            #code.interact(local=locals())

            #print(self.spec_col_label_img_id)

            #code.interact(local=locals())
            self.spec_col_label_img_ind += dir

            new_ind = self.spec_col_idxs[self.spec_col_label_img_ind]

            #ugly solution, improve later
            #really what I want to do is store a list of all where spec_col_label == true and then go up and down
           # if dir == 1:
            #    next_spec_col_label = new_ind
            #    count = 0
             #   while new_ind ==next_spec_col_label and self.spec_col_label_img_id+count < len(self.data):
            #        count+=1
            #        ser = (self.data.spec_col_label_id[0:self.spec_col_label_img_id + count] != "")
            #        new_ind = ser.where(ser).last_valid_index()

            #prev_ind = self.data.spec_col_label_id[0:self.spec_col_label_img_id].last_valid_index()

            #print(prev_ind)
            #print(type(prev_ind))

            if new_ind != None:
                self.refresh_spec_col_label(ind = new_ind)

    def refresh_spec_col_label(self,event='default',ind='default'):

        if not self.lock_spec_col_label:

            if ind == 'default':
                #print(self.data.spec_col_label_id)
                ser = (self.data.spec_col_label_id[0:self.cur_ind+1] != "")
                ind = ser.where(ser).last_valid_index()
                #ind = self.data.spec_col_label_id[0:self.cur_ind+1].last_valid_index()

            if ind != None:

                #pull up latest spec_col_label based on current id.
                spec_col_label_path = self.data["image_croppath"][ind]
                spec_col_label = Image.open(spec_col_label_path)
                spec_col_label = spec_col_label.resize((300, 300))  # Adjust the size as needed
                spec_col_label_tk = ImageTk.PhotoImage(spec_col_label)

                self.spec_col_label_label.config(image=spec_col_label_tk)
                self.spec_col_label_label.image = spec_col_label_tk

                #retain memory of which spec_col_label is currently displayed.
                self.spec_col_label_img_id = deepcopy(ind)

    def populate_from_spec_col_label(self,event="default"):

        #populate based on the index of the current spec_col_label.

        self.refresh_data(self.spec_col_label_img_id)

        # this will fill in current values with those matching the current spec_col_label in the spec_col_label pane.

    def populate_from_last(self,event="default"):

        #this will take from the last spec_col_label that matches current one.

        if self.cur_ind !=0:
            self.refresh_data(self.cur_ind-1)

        # this will fill in current values with those from the very last full image

    def clear_data(self,event="default"):

        [field.entry.delete(0, tk.END) for field in self.fields]

    #if I run into performance issues, I can try feeding chunks of images to pipeline, perhaps asynchronously
    def run(self,mode):
        #run these methods on different threads

        #thread 1: disable for now to let imshow work properly
        #Thread(target=self.process_images).start()

        #self.process_images()

        #thread 2:
        #thread2 = self.save_sometimes()

        #thread 3: let user visualize data
        #wait for data to appear before loading up gui
        #while(len(self.data==0)):
        #    sleep(5)

        #prior to working out multithread and live updating
        if mode == 'review':
            self.review_data()
        elif mode == 'process':
            self.process_images()
            self.review_data()


#complete version of this should allow user to either process data, review data, or do both.

Program(IMAGEFILES,DATAPATH,DATA_KEYS,FIELD_NAMES_APP,FIELD_NAMES_DATA).run(mode)


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