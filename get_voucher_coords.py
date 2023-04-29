from keras_ocr.tools import read
import code
import cv2
import numpy as np
from time import time
import random
import pytesseract

#use keras ocr to locate text in image
#use semantic similarity algorithm to locate title text
#try 4 times for different image rotations if not working
#return bounding box of text, then use the bounding box of text to predict voucher coordinates

#semantic similarity: don't need to use ml, instead, I will export a score based on length similarity
#and letter similarity

def simple_semantic(word1,word2):
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


#return either coords or none

class ImageToCoords:

    def __init__(self,kocr,image,voucher_dims,voucher_dims_dict):
        self.image_name = image
        self.image = cv2.imread(image)
        self.kocr = kocr
        self.voucher_dims = voucher_dims
        self.voucher_dims_dict = voucher_dims_dict.copy()
        self.semvecs_threshold = 1
        self.indeces_dict = {}
        self.rotate_count = 0

    def predict(self,image):
        return(self.kocr.recognize([read(image)]))

    def rotate_img(self):
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.rotate_count += 1
    # return list of indeces that correspond to a match, or return none

    def get_coords(self):

        self.predictions = self.predict(self.image)

        #code.interact(local=locals())
        #assess semantic similarity of each word
        if self.predictions != [[]]:

            self.labs = [f[0] for f in self.predictions[0]]

            #populate dictionary of which labels correspond to which label index.
            for m in self.voucher_dims_dict["labels"]:
                semantic_vec = []
                for p in self.labs:
                    semantic_vec.append(simple_semantic(m, p))
                    semantic_vec_min = min(semantic_vec)
                    #check that min is unique, and that min is below threshold
                    if semantic_vec.count(semantic_vec_min) == 1 and semantic_vec_min <= self.semvecs_threshold:
                        self.indeces_dict[m] = semantic_vec.index(semantic_vec_min)

            if self.indeces_dict == {}:
                if self.rotate_count < 3:
                    self.rotate_img()
                    self.get_coords()
                else:
                    return None
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

                #loop through each field and crop:

                for m in self.voucher_dims_dict["fields"]:
                    img_crop = im_out[round(self.voucher_dims_dict['fields'][m][0][1]):round(self.voucher_dims_dict['fields'][m][2][1]),
                                          round(self.voucher_dims_dict['fields'][m][0][0]):round(self.voucher_dims_dict['fields'][m][1][0])]

                    #rcol = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    #for p in range(3):
                        # code.interact(local=locals())
                    #    cv2.line(im_out, (round(self.voucher_dims_dict['fields'][m][p][0]),
                    #                      round(self.voucher_dims_dict['fields'][m][p][1])),
                    #             (round(self.voucher_dims_dict['fields'][m][(p + 1) % 4][0]),
                    #              round(self.voucher_dims_dict['fields'][m][(p + 1) % 4][1])), color=rcol,
                    #             thickness=4)
                    #now, I can actually just test out other text recognition on the available fields. Try keras ocr again to start.

                    #pred_label = self.predict(img_crop)

                    #also want to save the crop ultimately, so that I can possible train a custom model later.
                    #code.interact(local=locals())
                    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

                    #_,thresh = cv2.threshold(gray,175,255,cv2.THRESH_BINARY_INV)

                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                    #hresh = cv2.cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C

                    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)



                    #col = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
                    #pred_label = self.predict(col)

                    #this is hot garbage, still. next to try is the huggingface model.
                    #I don't think it is worth to try individual segmentation of numbers/letters... too much letter
                    #overlap in the small fields.
                    pred_label = pytesseract.image_to_string(sharpened,config="--psm 8 -c tessedit_char_whitelist=0123456789") #this is for only numbers -c tessedit_char_whitelist=0123456789

                    col = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
                    print(self.predict(col))
                    print(pred_label)
                    #code.interact(local=locals())
                    #self.voucher_dims_dict["fields"][m]["pred_label"] = pred_label

                    #cv2.imshow(str(pred_label), sharpened)
                    #cv2.waitKey(500)

                    cv2.imwrite("testout/" + str(time()) + "_" + str(pred_label) + ".jpg", sharpened)



                #cv2.imwrite(f"testout/{time()} full_img.jpg", im_out)

                    #cv2.imwrite(f"testout/{time()} cropped_img.jpg", img_crop)

                #im_out = cv2.resize(im_out, (600, 800))
                #cv2.imshow('image', im_out)
                #cv2.waitKey(0)

                #im_out = cv2.resize(im_out, (600, 800))

                #file = open(f"testout/{time.time()} preds.txt", 'w')
                #json.dump(self.predictions[0], file)
                #file.close()
                #cv2.imshow('image', image2)
                #cv2.moveWindow('image', 40, 30)
                #cv2.waitKey(0)
                        #block to visualize field crops
                #image2 = cv2.resize(self.image, (300, 400))
                #image2[round(form_tl_ys_avg / 10), round(form_tl_xs_avg / 10)] = [0, 0, 255]
                #for m in self.voucher_dims_dict['fields_adj']:
                #    for p in range(4):
                #        image2[round(self.voucher_dims_dict['fields_adj'][m][p][1]/10),round(self.voucher_dims_dict['fields_adj'][m][p][0]/10)]=[0,0,255]

                #cv2.imshow('image', image2)
                #cv2.waitKey(0)



                #code.interact(local=locals())


                #cv2.waitKey(0)

                #now what? need to store cropped form image (on disk?) and retain path.
                #need to crop fields and store cropped images (use later for training), retain empty fields for data review

                #average all the above metrics, use for creation of new rectangle plus update field locations.
        else:
            return None

    #based on semantic similarity vector (use all three words to try to reference, assume fail if can't)
    #rotate and rerun until have rotated 4 times (at that point, return none)

    #use the identified labels, extract their bounding boxes, and form the combined bounding box.

    #compare the combined bounding box with that in dict to determine relative size and orientation

    #use relationship logic to calculate absolute predicted position of the voucher.



    #def get_semantic_vecs(self):
    #    self.semvecs = [[], [], []]
    #    for p in range(len(self.labs)):
    #        # code.interact(local=locals())
    #        self.semvecs[0].append(simple_semantic("Specimen", self.labs[p]))
    #        self.semvecs[1].append(simple_semantic("Collection", self.labs[p]))
    #        self.semvecs[2].append(simple_semantic("Label", self.labs[p]))

    #def pop_valid_indeces(self):

    #    self.indeces = []
    #    for n in range(len(self.semvecs_threshold)):
    #        print(min(self.semvecs[n]))
    #        self.indeces = self.indeces + [i for i in range(len(self.semvecs[n])) if self.semvecs[n][i] < self.semvecs_threshold[n]]