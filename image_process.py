from keras_ocr.tools import read
import code
import cv2
import numpy as np
from time import time
import pytesseract
import random
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt



#use keras ocr to locate text in image
#use lexical similarity algorithm to locate title text
#try 4 times for different image rotations if not working
#return bounding box of text, then use the bounding box of text to predict voucher coordinates

#lexical similarity: don't need to use ml, instead, I will export a score based on length similarity
#and letter similarity

def simple_lexical(word1,word2):
    word1 = word1.lower()
    word2 = word2.lower()
    length_res = abs(len(word1) - len(word2))
    diff_letter_count1_2 = 0
    for n in word1:
        if n not in word2:
            diff_letter_count1_2 += 1

    diff_letter_count2_1 = 0
    for n in word2:
        if n not in word1:
            diff_letter_count2_1 += 1

    diff_letter_count = (diff_letter_count1_2 + diff_letter_count2_1 )/2

    return length_res + diff_letter_count

def crop_text_region(img):
    lightmin = 255 - (img.max() / 2)

    w_on_b = (255 - img)

    # try to crop to just text region:

    # steps:

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilate = cv2.dilate(w_on_b, kernel, iterations=8)

    _, thresh = cv2.threshold(dilate, lightmin, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # crop by this
    # print()
    if len(cnts[0]) == 0:
        sharpened2 = np.empty(shape=(0, 0))
    else:
        x, y, w, h = cv2.boundingRect(cnts[0][0])
        sharpened2 = img[y:(y + h), x:(x + w)]

    return sharpened2

#picture similarity
#this doesn't work very well. I can do better!
def mse(img1, img2):
   h,w,_ = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def hist_metric(img1,img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist2 = cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    metric_val = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return metric_val

def iterate_on_match(id,cur_dict,ref_dict):

    metric_treshold = 0.925
    #matches = True
    #first, test if dicts contain smae labels.
    #turning this off, don't think it is necessary

    #for m in cur_dict:
    #    m_match = False
    #    for p in ref_dict:
    #        if m == p:
    #            m_match = True
    #            break

    #    if m_match == False:
    #        matches = False
    #        break

    #if matches == False:
    #    return id+1

    #then, check if content of dict are close enough to equal

    hist_metrics = []
    for m in cur_dict:
        if m in ref_dict:
            hist_metric_ = hist_metric(cur_dict[m],ref_dict[m])
            #print("metric:"  + str(hist_metric_))
            hist_metrics.append(hist_metric_)

    #print("mse_min:" + str(sum(hist_metrics) / len(hist_metrics)))
    if len(hist_metrics) > 0:
        if min(hist_metrics) > metric_treshold:
            return id

    return id + 1

#return either coords or none

class ImageToData:

    def __init__(self,kocr,tr_ocr_processor,trocr,image,voucher_dims,voucher_dims_dict,prev_array_dict,max_id,data,predict_fields):
        self.image_name = image
        self.image = cv2.imread(image)
        self.kocr = kocr
        self.trocr = trocr
        self.tr_ocr_processor = tr_ocr_processor
        self.voucher_dims = voucher_dims
        self.voucher_dims_dict = voucher_dims_dict.copy()
        self.semvecs_threshold = 1
        self.indeces_dict = {}
        self.rotate_count = 0
        self.prev_array_dict = prev_array_dict
        self.cur_array_dict ={}
        self.cur_id = max_id
        self.data = data
        self.predict_fields = predict_fields


    def predict(self,image):
        return(self.kocr.recognize([read(image)]))

    def field_predict(self,img):

        col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        #add border, or
        #image = cv2.copyMakeBorder(image, 0, 384-image.shape[0], 0, 384-image.shape[1], cv2.BORDER_CONSTANT)
        #resize
        image = cv2.resize(col, (384,384), interpolation = cv2.INTER_AREA)
        #both
        #code.interact(local=locals())
        #upscale_perc =384/image.shape[1]

        #image = cv2.resize(image, (int(image.shape[1]*upscale_perc), int(image.shape[0]*upscale_perc)), interpolation=cv2.INTER_AREA)
        #image = cv2.copyMakeBorder(image, 0, 384 - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)

        #code.interact(local=locals())

        transform = transforms.ToTensor()
        # Convert the image to PyTorch tensor
        tensor = transform(image)

        tensor = tensor[None,:]

        generated_ids = self.trocr.generate(tensor)
        generated_text = self.tr_ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #code.interact(local=locals())
        #
        return(generated_text)

    def process_field(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((1, 40), np.uint8)
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        dst = cv2.add(gray, (255 - morphed))

        # _,thresh = cv2.threshold(gray,175,255,cv2.THRESH_BINARY_INV)

        blurred = cv2.GaussianBlur(dst, (5, 5), 0)

        # hresh = cv2.cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C

        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

        # code.interact(local=locals())

        #
        # sharpened2 = crop_text_region(sharpened)
        sharpened2 = sharpened

        return sharpened2


    def rotate_img(self):
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.rotate_count += 1
    # return list of indeces that correspond to a match, or return none

    def get_data(self):

        self.predictions = self.predict(self.image)

        #code.interact(local=locals())
        #assess lexical similarity of each word
        if self.predictions != [[]]:

            self.labs = [f[0] for f in self.predictions[0]]

            #populate dictionary of which labels correspond to which label index.
            match_count = 0
            for m in self.voucher_dims_dict["labels"]:
                lexical_vec = []
                for p in self.labs:
                    lexical_vec.append(simple_lexical(m, p))
                    lexical_vec_min = min(lexical_vec)
                    #check that min is unique, and that min is below threshold
                    if lexical_vec.count(lexical_vec_min) == 1 and lexical_vec_min <= self.semvecs_threshold:
                        match_count += 1
                        self.indeces_dict[m] = lexical_vec.index(lexical_vec_min)
                        #print(self.predictions[0][self.indeces_dict[m]][0])
                        #code.interact(local=locals())
            #need 6 matches to treat as succesful
            #6 so that if form is cut off will not recieve it successfully.
            if match_count < 8:
                self.indeces_dict = {}

            if self.indeces_dict == {}:
                if self.rotate_count < 3:
                    self.rotate_img()
                    self.get_data()
                else:
                    #return empty data. set cur array as prev array
                    self.data["voucher_id"] = None
                    self.cur_array_dict = self.prev_array_dict
            else:

                image2 = self.image.copy()

               # code.interact(local=locals())

                #use homography to correct form.

                #get all discovered translated points and source points
                pts_translated =[]
                pts_source = []
                for i in self.indeces_dict:
                    index = self.indeces_dict[i]
                    pts_translated.append(self.predictions[0][index][1])

                    pts_source.append(np.array(self.voucher_dims_dict["labels"][i]))

                pts_translated = np.vstack(pts_translated)
                pts_source = np.vstack(pts_source)

                # Calculate Homography
                h, status = cv2.findHomography(pts_translated,pts_source)

                im_out = cv2.warpPerspective(image2, h, (image2.shape[1], image2.shape[0]))

                #im_out_cp = im_out.copy()

                img_full_crop = im_out[self.voucher_dims[0][1]:self.voucher_dims[2][1],self.voucher_dims[0][0]:self.voucher_dims[1][0]]
                #code.interact(local=locals())
                path = f"training_data/{self.data['id']}"
                if not os.path.exists(path):
                    os.mkdir(path)
                img_name = path + "/cropped_form.png"
                cv2.imwrite(img_name, img_full_crop)

                self.data["image_croppath"] = os.getcwd() + '/' + img_name

                #loop through each field and crop:
                #should I make this smarter- only go through fields which have a label to match? Don't even have that relation right now...

                for m in self.voucher_dims_dict["fields"]:
                    img_field = im_out[round(self.voucher_dims_dict['fields'][m][0][1]):round(self.voucher_dims_dict['fields'][m][2][1]),
                                          round(self.voucher_dims_dict['fields'][m][0][0]):round(self.voucher_dims_dict['fields'][m][1][0])]

                    img_crop_processed = self.process_field(img_field)

                    cv2.imwrite(path + f"/{m}.png",img_crop_processed)

                    self.cur_array_dict[m] = img_crop_processed

                    if self.predict_fields:
                        prediction = self.field_predict(img_crop_processed)
                    else:
                        prediction = None

                    self.data["pred_" + m] = prediction


                #compare current arrays to previous. if they match, do not increment id, otherwise, increment
                if self.cur_array_dict != {}:
                    newid = iterate_on_match(self.cur_id,self.cur_array_dict,self.prev_array_dict)

                    self.data["voucher_id"] = newid

                else:
                    self.data["voucher_id"] = None

                    self.cur_array_dict = self.prev_array_dict

                #return self.pred_data,self.cur_array_dict

        #if skips the current image, return prev_array_dict instead of cur_array_dict

        return self.data, self.cur_array_dict

    #based on lexical similarity vector (use all three words to try to reference, assume fail if can't)
    #rotate and rerun until have rotated 4 times (at that point, return none)

    #use the identified labels, extract their bounding boxes, and form the combined bounding box.

    #compare the combined bounding box with that in dict to determine relative size and orientation

    #use relationship logic to calculate absolute predicted position of the voucher.



    #def get_lexical_vecs(self):
    #    self.semvecs = [[], [], []]
    #    for p in range(len(self.labs)):
    #        # code.interact(local=locals())
    #        self.semvecs[0].append(simple_lexical("Specimen", self.labs[p]))
    #        self.semvecs[1].append(simple_lexical("Collection", self.labs[p]))
    #        self.semvecs[2].append(simple_lexical("Label", self.labs[p]))

    #def pop_valid_indeces(self):

    #    self.indeces = []
    #    for n in range(len(self.semvecs_threshold)):
    #        print(min(self.semvecs[n]))
    #        self.indeces = self.indeces + [i for i in range(len(self.semvecs[n])) if self.semvecs[n][i] < self.semvecs_threshold[n]]

    # cv2.imwrite(f"testout/{time()} full_img.jpg", im_out)

    # cv2.imwrite(f"testout/{time()} cropped_img.jpg", img_crop)

    # im_out = cv2.resize(im_out, (600, 800))
    # cv2.imshow('image', im_out)
    # cv2.waitKey(0)

    # im_out = cv2.resize(im_out, (600, 800))

    # file = open(f"testout/{time.time()} preds.txt", 'w')
    # json.dump(self.predictions[0], file)
    # file.close()
    # cv2.imshow('image', image2)
    # cv2.moveWindow('image', 40, 30)
    # cv2.waitKey(0)
    # block to visualize field crops
    # image2 = cv2.resize(self.image, (300, 400))
    # image2[round(form_tl_ys_avg / 10), round(form_tl_xs_avg / 10)] = [0, 0, 255]
    # for m in self.voucher_dims_dict['fields_adj']:
    #    for p in range(4):
    #        image2[round(self.voucher_dims_dict['fields_adj'][m][p][1]/10),round(self.voucher_dims_dict['fields_adj'][m][p][0]/10)]=[0,0,255]

    # cv2.imshow('image', image2)
    # cv2.waitKey(0)

    # code.interact(local=locals())

    # cv2.waitKey(0)

    # now what? need to store cropped form image (on disk?) and retain path.
    # need to crop fields and store cropped images (use later for training), retain empty fields for data review

    # average all the above metrics, use for creation of new rectangle plus update field locations.

    # write out cropped image so it can easily be located later
    # do this at the end so I get it with any augmentations

    # now, I can actually just test out other text recognition on the available fields. Try keras ocr again to start.

    # pred_label = self.predict(img_crop)

    # also want to save the crop ultimately, so that I can possible train a custom model later.
    # code.interact(local=locals())

    # discard v small fields
    # if sharpened2.shape[0]>15 and sharpened2.shape[1]>15:

    # print(sharpened2)

    # code.interact(local=locals())

    # col = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    # pred_label = self.predict(col)

    # this is hot garbage, still. next to try is the huggingface model.
    # I don't think it is worth to try individual segmentation of numbers/letters... too much letter
    # overlap in the small fields.
    # tempname = "testout/" + str(time()) + ".jpg"

    # print("name " + tempname)

    # print("tesseract result:")
    # pred_label = pytesseract.image_to_string(sharpened2,config="--psm 8 -c tessedit_char_whitelist=0123456789") #-c tessedit_char_whitelist=0123456789") #this is for only numbers -c tessedit_char_whitelist=0123456789
    # pred_label = pytesseract.image_to_string(sharpened2,config="digits")
    # pred_label = pytesseract.image_to_string(sharpened)
    # print(pred_label)

    # col = cv2.cvtColor(sharpened2, cv2.COLOR_GRAY2BGR)

    # code.interact(local=locals())
    # print('keras ocr result:')
    # outlabs =self.predict(col)[0]

    # if len(outlabs)==0:
    #    print(outlabs)
    # else:
    #    print(outlabs[0][0])

    # print("trocr result:")
    # code.interact(local=locals())
    # trocr_out = self.predict_trocr(col)
    # print(trocr_out) #torch.from_numpy(

    # save the predicted values as data

    # write the array to dictionary
    # code.interact(local=locals())
    # self.data["pred_" + m] = trocr_out
    # self.cur_array_dict[m]=col

    # self.voucher_dims_dict["fields"][m]["pred_label"] = pred_label

    # cv2.imshow("out", sharpened)
    # cv2.waitKey(500)

    # write out each cropped field
    # cv2.imwrite(tempname, col)

    # else:
    #    cv2.imwrite(tempname, sharpened2)



    #draw boxes around cropped form:

    # code.interact(local=locals())
    # rcol = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # for m in self.voucher_dims_dict["fields"]:
    #    for p in range(3):
    #        cv2.line(im_out_cp, (round(self.voucher_dims_dict['fields'][m][p][0]),
    #                          round(self.voucher_dims_dict['fields'][m][p][1])),
    #                 (round(self.voucher_dims_dict['fields'][m][(p + 1) % 4][0]),
    #                  round(self.voucher_dims_dict['fields'][m][(p + 1) % 4][1])), color=rcol,
    #                 thickness=4)