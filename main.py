import os
import pandas as pd
from misc import is_img
from image_process import ImageToData
import keras_ocr
import matplotlib.pyplot as plt
from threading import Thread
from time import sleep
import tkinter as tk
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from copy import deepcopy
from PIL import ImageTk, Image
import code

#load trocr model

#image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

#pixel_values = processor(images=image, return_tensors="pt").pixel_values





pytesseract.pytesseract.tesseract_cmd = 'C:/Users/daniel.woodrich/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

#voucher dimensions (10px):
#this will be used to reference relative coordinates of different fields
#VOUCHER_DIMS = [[0,0],[118,0],[0,95],[118,95]]
VOUCHER_DIMS = [[0,0],[442,0],[0,352],[442,352]]

#make this more customizable later
IMAGEDIR = "Y:/RACE_Imagery/Field_Photos/EBSshelf2022/Vesteraalen/Leg3"
#IMAGEDIR = "Y:/RACE_Imagery/Field_Photos/EBSshelf2021/Vesteraalen/Leg3/EBS"
#IMAGEDIR =  "C:/Users/daniel.woodrich/Work/Data mgmt coord/Projects/Race_data_reco/testphotos"
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

#data will be stored in a table- read this from file at script start.
DATAPATH = "data.csv"
DATA_KEYS = {"id":None,"voucher_id":None,"image_fullpath":None,"image_croppath":None,"pred_vessel":None,"pred_cruise_number":None,
             "pred_haul_number":None,"pred_specimen_number":None,"pred_stomach_sample":None,"pred_tissue_sample":None,"pred_right_ovary":None,
            "pred_left_ovary":None,"pred_whole_animal":None,"pred_length_cm":None,"pred_weight_gm":None,"pred_species_identification":None,
            "pred_comments":None,"pred_collector_initials":None,"pred_preservative":None,"verified_vessel":None,"verified_cruise_number":None,"verified_haul_number":None,
            "verified_specimen_number":None,"verified_stomach_sample":None,"verified_tissue_sample":None,"verified_right_ovary":None,"verified_left_ovary":None,
            "verified_whole_animal":None,"verified_length_cm":None,	"verified_weight_gm":None,"verified_species_identification":None,"verified_comments":None,
            "verified_collector_initials":None,	"verified_preservative":None
}

#parameter to turn on and off label prediction
PREDICT_FIELDS = False

class Program:

    #can make this more specific to describe images needing processed as well as analyzed.
    def __init__(self,images_in,data_in,datakeys):
        #once have some data, do
        self.datakeys = datakeys
        self.datapath = data_in
        #code.interact(local=locals())
        self.data = pd.read_csv(data_in)
        if self.data.empty:
            self.image_to_go = images_in
        else:
            #code.interact(local=locals())
            self.image_to_go = [f for f in images_in if f not in self.data.image_fullpath.to_list()]
        self.processed_image_coords = {} #key is image name, each is a dict of 1 coords and 2 path to image crop.
        self.reviewed_image_data = {} #
        self.prev_array ={}
        #populate below by looking in loaded data
        #code.interact(local=locals())
        self.max_voucher_id = self.data.voucher_id.dropna().max()
        self.max_row_id = self.data.id.dropna().max()

        #test for NA, give 0 if na
        if self.max_voucher_id != self.max_voucher_id:
            #code.interact(local=locals())
            self.max_voucher_id = 1

        # test for NA, give 0 if na
        if self.max_row_id != self.max_row_id:
            self.max_row_id = 1


    def process_images(self):
        self.kocr = keras_ocr.pipeline.Pipeline()
        self.tr_ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.trocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        for n in range(len(self.image_to_go)):
            self.process_image(n)

    def process_image(self,i):

        #run the pipeline on a single image. output data into self.data
        focal_img = self.image_to_go[i]

        print(focal_img)

        #populates data needed for review method
        print(self.max_voucher_id)

        datain = deepcopy(self.datakeys)
        datain["id"] = self.max_row_id
        datain["image_fullpath"] = focal_img

        imagetodata= ImageToData(self.kocr,self.tr_ocr_processor,self.trocr,focal_img,VOUCHER_DIMS,VOUCHER_DIMS_DICT,self.prev_array,self.max_voucher_id,datain,PREDICT_FIELDS)

        #do it
        data,prev_array = imagetodata.get_data()

        if not data["voucher_id"] == None:
            self.max_voucher_id = data["voucher_id"]

        self.prev_array = prev_array

        self.max_row_id = self.max_row_id + 1

        row = pd.DataFrame(data, index=[0, ])

        row.to_csv(self.datapath, mode='a',index=False,header=False)

        #save data to csv. if max_voucher_id is none, change it to NA (no voucher present). still good to document
        #for future discovery.




        #after this runs, register image as having been completed.

        #from this initiate two proceses. 1- loop through fields and send to hugging face algo to
        #predict data present. save these in data, with reviewed column set to 0

        #also, have different threaded process running to present data ready (reviewed column 0) to review to user
    #kind of annoying but will have to independently assign all tkinter label methods in this style.
    def review_data(self):

        #use verified vessel as a proxy
        self.cur_ind = self.data.verified_vessel.isna().idxmin()-1

        #self.cur_encountered = []

        # code.interact(local=locals())
        #self.image_advance()

        self.lock_voucher = False

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
        self.full_image_label.pack()

        last_cropped_image_button = tk.Button(cropped_buttons_frame, text="Previous voucher",
                                              command=self.go_to_last_voucher)
        last_cropped_image_button.grid(row=0, column=0)

        self.refresh_button = tk.Button(cropped_buttons_frame, text="Latest voucher", command=self.refresh_voucher)
        self.refresh_button.grid(row=0, column=1)

        self.lock_button = tk.Button(cropped_buttons_frame, text="Lock voucher", command=self.toggle_lock)
        self.lock_button.grid(row=1, column=0)

        self.populate_button = tk.Button(cropped_buttons_frame, text="Populate from voucher", command=self.populate_from_voucher)
        self.populate_button.grid(row=1, column=1)

        self.voucher_label = tk.Label(top_right_frame)
        self.voucher_label.pack()

        fields_frame = tk.Frame(top_right_frame)
        fields_frame.pack()

        # Create the static labels
        #voucher_id_label = tk.Label(top_right_frame, text="Voucher ID:")
        #species_name_label = tk.Label(top_right_frame, text="Species Name:")
        self.vessel_label = tk.Label(fields_frame, text="VESSEL")
        self.cruise_number_label = tk.Label(fields_frame, text="CRUISE NUMBER")
        self.haul_number_label = tk.Label(fields_frame, text="HAUL NUMBER")
        self.specimen_number_label = tk.Label(fields_frame, text="SPECIMEN NUMBER")
        self.stomach_sample_label = tk.Label(fields_frame, text="STOMACH SAMPLE")
        self.tissue_sample_label = tk.Label(fields_frame, text="TISSUE SAMPLE")
        self.right_ovary_label = tk.Label(fields_frame, text="RIGHT OVARY")
        self.left_ovary_label = tk.Label(fields_frame, text="LEFT OVARY")
        self.whole_animal_label = tk.Label(fields_frame, text="WHOLE ANIMAL")
        self.length_cm_label = tk.Label(fields_frame, text="LENGTH (CM)")
        self.weight_gm_label = tk.Label(fields_frame, text="WEIGHT (GM)")
        self.species_identification_label = tk.Label(fields_frame, text="SPECIES IDENTIFICATION")
        self.comments_label = tk.Label(fields_frame, text="COMMENTS")
        self.collector_initials_label = tk.Label(fields_frame, text="COLLECTOR'S INITIALS")
        self.preservative_label = tk.Label(fields_frame, text="PRESERVATIVE")
        self.vessel_label.grid(row=0, column=0, sticky="w")
        self.cruise_number_label.grid(row=1, column=0, sticky="w")
        self.haul_number_label.grid(row=2, column=0, sticky="w")
        self.specimen_number_label.grid(row=3, column=0, sticky="w")
        self.stomach_sample_label.grid(row=4, column=0, sticky="w")
        self.tissue_sample_label.grid(row=5, column=0, sticky="w")
        self.right_ovary_label.grid(row=6, column=0, sticky="w")
        self.left_ovary_label.grid(row=7, column=0, sticky="w")
        self.whole_animal_label.grid(row=8, column=0, sticky="w")
        self.length_cm_label.grid(row=9, column=0, sticky="w")
        self.weight_gm_label.grid(row=10, column=0, sticky="w")
        self.species_identification_label.grid(row=11, column=0, sticky="w")
        self.comments_label.grid(row=12, column=0, sticky="w")
        self.collector_initials_label.grid(row=13, column=0, sticky="w")
        self.preservative_label.grid(row=14, column=0, sticky="w")

        # Create the entry fields
        self.vessel_entry = tk.Entry(fields_frame)
        self.cruise_number_entry  = tk.Entry(fields_frame)
        self.haul_number_entry = tk.Entry(fields_frame)
        self.specimen_number_entry  = tk.Entry(fields_frame)
        self.stomach_sample_entry = tk.Entry(fields_frame)
        self.tissue_sample_entry  = tk.Entry(fields_frame)
        self.right_ovary_entry = tk.Entry(fields_frame)
        self.left_ovary_entry  = tk.Entry(fields_frame)
        self.whole_animal_entry = tk.Entry(fields_frame)
        self.length_cm_entry  = tk.Entry(fields_frame)
        self.weight_gm_entry = tk.Entry(fields_frame)
        self.species_identification_entry  = tk.Entry(fields_frame)
        self.comments_entry = tk.Entry(fields_frame)
        self.collector_initials_entry  = tk.Entry(fields_frame)
        self.preservative_entry = tk.Entry(fields_frame)
        self.vessel_entry.grid(row=0, column=1, padx=5)
        self.cruise_number_entry.grid(row=1, column=1, padx=5)
        self.haul_number_entry.grid(row=2, column=1, padx=5)
        self.specimen_number_entry.grid(row=3, column=1, padx=5)
        self.stomach_sample_entry.grid(row=4, column=1, padx=5)
        self.tissue_sample_entry.grid(row=5, column=1, padx=5)
        self.right_ovary_entry.grid(row=6, column=1, padx=5)
        self.left_ovary_entry.grid(row=7, column=1, padx=5)
        self.whole_animal_entry.grid(row=8, column=1, padx=5)
        self.length_cm_entry.grid(row=9, column=1, padx=5)
        self.weight_gm_entry.grid(row=10, column=1, padx=5)
        self.species_identification_entry.grid(row=11, column=1, padx=5)
        self.comments_entry.grid(row=12, column=1, padx=5)
        self.collector_initials_entry.grid(row=13, column=1, padx=5)
        self.preservative_entry.grid(row=14, column=1, padx=5)

        data_buttons_frame = tk.Frame(top_right_frame)
        data_buttons_frame.pack()

        self.from_last_button = tk.Button(data_buttons_frame, text="Last", command=self.populate_from_last)
        self.from_last_button.grid(row=0, column=0, padx=5)

        self.clear_data_button = tk.Button(data_buttons_frame, text="Clear", command=self.clear_data)
        self.clear_data_button.grid(row=0, column=1, padx=5)

        self.save_button = tk.Button(data_buttons_frame, text="Save", command=self.save_data)
        self.save_button.grid(row=0, column=2, padx=5)

        window.bind("<Down>", lambda e: self.image_cycle(e,-1))
        window.bind("<Up>", lambda e: self.image_cycle(e,1))
        window.bind("<Return>", self.save_data)

        window.bind("<Control-s>",self.hard_save_data)
        window.bind("<Control-r>", self.populate_from_last)
        window.bind("<Control-p>", self.populate_from_voucher)

        window.bind("<Control-Left>", self.go_to_last_voucher)
        window.bind("<Control-Right>", self.refresh_voucher)
        window.bind("<Control-l>", self.toggle_lock)
        window.bind("<Control-w>", self.clear_data)

        self.image_cycle("start",1)

        window.mainloop()

    def refresh_full_image(self):
        image_path = self.data["image_fullpath"][self.cur_ind]  # Replace with your image path
        full_image = Image.open(image_path)
        full_image = full_image.resize((450, 450))  # Adjust the size as needed
        full_image_tk = ImageTk.PhotoImage(full_image)

        self.full_image_label.config(image=full_image_tk)
        self.full_image_label.image = full_image_tk

    def refresh_data(self,ind):

        self.save_button.config(text="Save")

        self.cur_data = self.data.iloc[[ind]].reset_index(drop=True)
        #print(self.cur_data)

        #if(self.cur_ind!=0):
        #    code.interact(local=locals())
        #

        self.vessel_entry.delete(0, tk.END)
        if self.cur_data["verified_vessel"].isna().bool():
            if self.cur_data["pred_vessel"].isna().bool():
                self.vessel_entry.insert(0,"")
            else:
                self.vessel_entry.insert(0, self.cur_data.loc[0,"pred_vessel"])
        else:
            self.vessel_entry.insert(0, self.cur_data.loc[0,"verified_vessel"])

        self.cruise_number_entry.delete(0, tk.END)
        if self.cur_data["verified_cruise_num"].isna().bool():
            if self.cur_data["pred_cruise_num"].isna().bool():
                self.cruise_number_entry.insert(0,"")
            else:
                self.cruise_number_entry.insert(0, self.cur_data.loc[0,"pred_cruise_num"])
        else:
            self.cruise_number_entry.insert(0, self.cur_data.loc[0,"verified_cruise_num"])

        self.haul_number_entry.delete(0, tk.END)
        if self.cur_data["verified_haul_num"].isna().bool():
            if self.cur_data["pred_haul_num"].isna().bool():
                self.haul_number_entry.insert(0,"")
            else:
                self.haul_number_entry.insert(0, self.cur_data.loc[0,"pred_haul_num"])
        else:
            self.haul_number_entry.insert(0, self.cur_data.loc[0,"verified_haul_num"])

        self.specimen_number_entry.delete(0, tk.END)
        if self.cur_data["verified_specimen_num"].isna().bool():
            if self.cur_data["pred_specimen_num"].isna().bool():
                self.specimen_number_entry.insert(0, "")
            else:
                self.specimen_number_entry.insert(0, self.cur_data.loc[0,"pred_specimen_num"])
        else:
            self.specimen_number_entry.insert(0, self.cur_data.loc[0,"verified_specimen_num"])

        self.stomach_sample_entry.delete(0, tk.END)
        if self.cur_data["verified_stomach_sample"].isna().bool():
            if self.cur_data["pred_stomach_sample"].isna().bool():
                self.stomach_sample_entry.insert(0, "")
            else:
                self.stomach_sample_entry.insert(0, self.cur_data.loc[0,"pred_stomach_sample"])
        else:
            self.stomach_sample_entry.insert(0, self.cur_data.loc[0,"verified_stomach_sample"])

        self.tissue_sample_entry.delete(0, tk.END)
        if self.cur_data["verified_tissue_sample"].isna().bool():
            if self.cur_data["pred_tissue_sample"].isna().bool():
                self.tissue_sample_entry.insert(0, "")
            else:
                self.tissue_sample_entry.insert(0, self.cur_data.loc[0,"pred_tissue_sample"])
        else:
            self.tissue_sample_entry.insert(0, self.cur_data.loc[0,"verified_tissue_sample"])

        self.right_ovary_entry.delete(0, tk.END)
        if self.cur_data["verified_right_ovary"].isna().bool():
            if self.cur_data["pred_right_ovary"].isna().bool():
                self.right_ovary_entry.insert(0, "")
            else:
                self.right_ovary_entry.insert(0, self.cur_data.loc[0,"pred_right_ovary"])
        else:
            self.right_ovary_entry.insert(0, self.cur_data.loc[0,"verified_right_ovary"])

        self.left_ovary_entry.delete(0, tk.END)
        if self.cur_data["verified_left_ovary"].isna().bool():
            if self.cur_data["pred_left_ovary"].isna().bool():
                self.left_ovary_entry.insert(0, "")
            else:
                self.left_ovary_entry.insert(0, self.cur_data.loc[0,"pred_left_ovary"])
        else:
            self.left_ovary_entry.insert(0, self.cur_data.loc[0,"verified_left_ovary"])

        self.whole_animal_entry.delete(0, tk.END)
        if self.cur_data["verified_whole_animal"].isna().bool():
            if self.cur_data["pred_whole_animal"].isna().bool():
                self.whole_animal_entry.insert(0, "")
            else:
                self.whole_animal_entry.insert(0, self.cur_data.loc[0,"pred_whole_animal"])
        else:
            self.whole_animal_entry.insert(0, self.cur_data.loc[0,"verified_whole_animal"])

        self.length_cm_entry.delete(0, tk.END)
        if self.cur_data["verified_length_cm"].isna().bool():
            if self.cur_data["pred_length_cm"].isna().bool():
                self.length_cm_entry.insert(0, "")
            else:
                self.length_cm_entry.insert(0, self.cur_data.loc[0,"pred_length_cm"])
        else:
            self.length_cm_entry.insert(0, self.cur_data.loc[0,"verified_length_cm"])

        self.weight_gm_entry.delete(0, tk.END)
        if self.cur_data["verified_weight_gm"].isna().bool():
            if self.cur_data["pred_weight_gm"].isna().bool():
                self.weight_gm_entry.insert(0, "")
            else:
                self.weight_gm_entry.insert(0, self.cur_data.loc[0,"pred_weight_gm"])
        else:
            self.weight_gm_entry.insert(0, self.cur_data.loc[0,"verified_weight_gm"])

        self.species_identification_entry.delete(0, tk.END)
        if self.cur_data["verified_species_identification"].isna().bool():
            if self.cur_data["pred_species_identification"].isna().bool():
                self.species_identification_entry.insert(0, "")
            else:
                self.species_identification_entry.insert(0, self.cur_data.loc[0,"pred_species_identification"])
        else:
            self.species_identification_entry.insert(0, self.cur_data.loc[0,"verified_species_identification"])

        self.comments_entry.delete(0, tk.END)
        if self.cur_data["verified_comments"].isna().bool():
            if self.cur_data["pred_comments"].isna().bool():
                self.comments_entry.insert(0, "")
            else:
                self.comments_entry.insert(0, self.cur_data.loc[0,"pred_comments"])
        else:
            self.comments_entry.insert(0, self.cur_data.loc[0,"verified_comments"])

        self.collector_initials_entry.delete(0, tk.END)
        if self.cur_data["verified_collector_initials"].isna().bool():
            if self.cur_data["pred_collector_initials"].isna().bool():
                self.collector_initials_entry.insert(0, "")
            else:
                self.collector_initials_entry.insert(0, self.cur_data.loc[0,"pred_collector_initials"])
        else:
            self.collector_initials_entry.insert(0, self.cur_data.loc[0,"verified_collector_initials"])

        self.preservative_entry.delete(0, tk.END)
        if self.cur_data["verified_preservative"].isna().bool():
            if self.cur_data["pred_preservative"].isna().bool():
                self.preservative_entry.insert(0, "")
            else:
                self.preservative_entry.insert(0, self.cur_data.loc[0,"pred_preservative"])
        else:
            self.preservative_entry.insert(0, self.cur_data.loc[0,"verified_preservative"])

    def image_cycle(self,event='default',direction = 1):

        #populate the full image and the new row of data.
        #populate verified cols that are not na, if na try predicted cols.
        #print(event)

        self.cur_ind = self.cur_ind + direction

        if self.cur_ind < 0 or self.cur_ind > len(self.data)-1:
            self.cur_ind = self.cur_ind - direction
        else:

            self.refresh_full_image()

            #test if image is predicted to be same voucher, if so preload it. only do it in forward direction.
            if self.cur_ind > 0 and direction == 1: #and self.cur_ind not in self.cur_encountered
                #print(self.data.loc[self.cur_ind-1,"voucher_id"])
                #print(self.data.loc[self.cur_ind, "voucher_id"])
                if self.data.loc[self.cur_ind-1,"voucher_id"] == self.data.loc[self.cur_ind,"voucher_id"] and pd.isna(self.data.loc[self.cur_ind,"verified_vessel"]):
                    self.refresh_data(self.cur_ind-1)
                    self.save_button.config(text="Autofilled... Save")
                else:
                    self.refresh_data(self.cur_ind)
            else:
                self.refresh_data(self.cur_ind)

            if not self.lock_voucher:
                #code.interact(local=locals())
                #self.data["voucher_id"].iloc[[self.cur_ind]].equals(self.data["voucher_id"].iloc[[self.cur_ind-1]]) and
                if not self.data["voucher_id"].iloc[[self.cur_ind]].isna().bool():
                    #print("didit")
                    self.refresh_voucher("automatic",self.cur_ind)

        #self.cur_encountered.append(self.cur_ind)

    def save_data(self,event="default"):

        self.data.loc[self.cur_ind,"verified_vessel"] = self.vessel_entry.get()
        self.data.loc[self.cur_ind,"verified_cruise_num"] = self.cruise_number_entry.get()
        self.data.loc[self.cur_ind, "verified_haul_num"]= self.haul_number_entry.get()
        self.data.loc[self.cur_ind, "verified_specimen_num"]= self.specimen_number_entry.get()
        self.data.loc[self.cur_ind,"verified_stomach_sample"] = self.stomach_sample_entry.get()
        self.data.loc[self.cur_ind, "verified_tissue_sample"]= self.tissue_sample_entry.get()
        self.data.loc[self.cur_ind,"verified_right_ovary"] = self.right_ovary_entry.get()
        self.data.loc[self.cur_ind, "verified_left_ovary"]= self.left_ovary_entry.get()
        self.data.loc[self.cur_ind,"verified_whole_animal"] = self.whole_animal_entry.get()
        self.data.loc[self.cur_ind, "verified_length_cm"]= self.length_cm_entry.get()
        self.data.loc[self.cur_ind,"verified_weight_gm"] = self.weight_gm_entry.get()
        self.data.loc[self.cur_ind, "verified_species_identification"]= self.species_identification_entry.get()
        self.data.loc[self.cur_ind, "verified_comments"]= self.comments_entry.get()
        self.data.loc[self.cur_ind,"verified_collector_initials"] = self.collector_initials_entry.get()
        self.data.loc[self.cur_ind, "verified_preservative"]= self.preservative_entry.get()

        #code.interact(local=locals())

        self.save_button.config(text="Saved!")

    def hard_save_data(self,event="default"):

        self.save_data()

        self.save_button.config(text="Hard Saved!")

        data_recent = pd.read_csv(self.datapath)

        data_out= pd.concat([self.data,data_recent[data_recent['id'].isin(self.data['id']) == False],])

        data_out.to_csv(self.datapath, index=False, header=True)

    def toggle_lock(self,event="default"):
        self.lock_voucher = not self.lock_voucher

        if self.lock_voucher:
            self.lock_button.config(relief=tk.SUNKEN)
        else:
            self.lock_button.config(relief=tk.RAISED)

    def go_to_last_voucher(self,event='default'):

        #find the last index where voucher id was diff and not na

        if not self.lock_voucher and not self.cur_ind <= 0 :
            #as of right now, dont use voucher id, just go back on position (that isn't NA)
            #code.interact(local=locals())

            prev_ind = self.data.voucher_id[0:self.voucher_img_id].last_valid_index()

            if prev_ind != None:
                self.refresh_voucher(ind = prev_ind)

    def refresh_voucher(self,event='default',ind='default'):

        if not self.lock_voucher:

            if ind == 'default':
                ind = self.data.voucher_id[0:self.cur_ind+1].last_valid_index()

            if ind != None:

                #pull up latest voucher based on current id.
                voucher_path = self.data["image_croppath"][ind]
                voucher = Image.open(voucher_path)
                voucher = voucher.resize((300, 300))  # Adjust the size as needed
                voucher_tk = ImageTk.PhotoImage(voucher)

                self.voucher_label.config(image=voucher_tk)
                self.voucher_label.image = voucher_tk

                #retain memory of which voucher is currently displayed.
                self.voucher_img_id = deepcopy(ind)

    def populate_from_voucher(self,event="default"):

        #populate based on the index of the current voucher.

        self.refresh_data(self.voucher_img_id)

        # this will fill in current values with those matching the current voucher in the voucher pane.

    def populate_from_last(self,event="default"):

        #this will take from the last voucher that matches current one.

        if self.cur_ind !=0:
            self.refresh_data(self.cur_ind-1)

        # this will fill in current values with those from the very last full image

    def clear_data(self,event="default"):

        self.vessel_entry.delete(0, tk.END)
        self.cruise_number_entry .delete(0, tk.END)
        self.haul_number_entry.delete(0, tk.END)
        self.specimen_number_entry .delete(0, tk.END)
        self.stomach_sample_entry.delete(0, tk.END)
        self.tissue_sample_entry .delete(0, tk.END)
        self.right_ovary_entry.delete(0, tk.END)
        self.left_ovary_entry .delete(0, tk.END)
        self.whole_animal_entry.delete(0, tk.END)
        self.length_cm_entry .delete(0, tk.END)
        self.weight_gm_entry.delete(0, tk.END)
        self.species_identification_entry .delete(0, tk.END)
        self.comments_entry.delete(0, tk.END)
        self.collector_initials_entry .delete(0, tk.END)
        self.preservative_entry.delete(0, tk.END)

    #if I run into performance issues, I can try feeding chunks of images to pipeline, perhaps asynchronously
    def run(self):
        #run these methods on different threads

        #thread 1: disable for now to let imshow work properly
        #thread1 = Thread(target=self.process_images).start()

        #self.process_images()

        #thread 2:
        #thread2 = self.save_sometimes()

        #thread 3: let user visualize data
        #wait for data to appear before loading up gui
        #while(len(self.data==0)):
        #    sleep(5)

        self.review_data()


#complete version of this should allow user to either process data, review data, or do both.

Program(IMAGEFILES,DATAPATH,DATA_KEYS).run()


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