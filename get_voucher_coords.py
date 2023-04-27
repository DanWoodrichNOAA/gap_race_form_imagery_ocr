from keras_ocr.tools import read
import code
import cv2
import math
import time #just for test, delete after
from statistics import median

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

#this is a stackoverflow function to get the cropped and straighted image from a rectangle at an angle.
def get_sub_image(rect, src):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)
    return out

def get_mod_field(coords,angle,exps_x, exps_y, form_tl_xs, form_tl_ys):

    #pseudo: find the modified tl of the field - the current coords describe
    #relative distance to the form tl. need to calculate original angle, then
    #add in angle and solve right triangle to find new tl coordinate.

    hyp = math.sqrt((coords[0][0]) ** 2 + (coords[0][1]) ** 2)

    og_angle = math.asin(coords[0][1]/ hyp)

    new_angle = og_angle - angle

    tlx = hyp * math.cos(new_angle)
    tly = hyp * math.sin(new_angle)

    #now, take these values, expand by x and y exps, and add to form tl to get absolute position

    tlx = (tlx * exps_x) + form_tl_xs
    tly = (tly * exps_y) + form_tl_ys

    x_hyp = (coords[1][0]-coords[0][0]) * exps_x
    y_hyp = (coords[2][1]-coords[1][1]) * exps_y

    trx = tlx + x_hyp * math.cos(angle)
    try_ = tly - x_hyp * math.sin(angle)

    blx = tlx + y_hyp * math.sin(angle)
    bly = tly + y_hyp * math.cos(angle)

    brx = blx + x_hyp * math.cos(angle)
    bry = bly - x_hyp * math.sin(angle)

    #coords_mod = [[tlx,tly],[trx,try_],[blx,bly],[brx,bry]]

    #instead of returning the above, want to try to return as cv2 rect type which
    #may allow me to crop it while rotated.

    centerx = (tlx + trx + blx + brx) / 4
    centery = (tly + try_ + bly + bry) / 4

    #cv2 expects degrees
    return ((centerx,centery),(x_hyp,y_hyp),math.degrees(angle))

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot
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

    def predict(self):
        return(self.kocr.recognize([read(self.image)]))

    def rotate_img(self):
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.rotate_count += 1
    # return list of indeces that correspond to a match, or return none

    def getTrigMetrics(self):
        angles = [] #angle of box relative to x
        exps_x = []
        exps_y = []
        form_tl_xs = []
        form_tl_ys = []
        for p in self.indeces_dict:
            box = self.predictions[0][self.indeces_dict[p]][1]
            # hyp = dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
            y_hyp = math.sqrt((box[3][0] - box[2][0]) ** 2 + (box[3][1] - box[2][1]) ** 2)
            bc = box[3][1] - box[2][1]

            #if bc != 0:
             #   code.interact(local=locals())

            # here is angle.
            angle = math.asin(bc / y_hyp)

            # now need expansion
            # take the avg of the ratio of current to template

            x_hyp = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

            yexpansion = y_hyp / (self.voucher_dims_dict["labels"][p][1][0] - self.voucher_dims_dict["labels"][p][0][0])
            xexpansion = x_hyp / (self.voucher_dims_dict["labels"][p][2][1] - self.voucher_dims_dict["labels"][p][1][1])

            # avg_exp = (yexpansion + xexpansion)/2

            # now using expansion, calculate expected position of tl corner of
            tl_adj = []
            tl_adj.append(self.voucher_dims_dict["labels"][p][0][0] * xexpansion)
            tl_adj.append(self.voucher_dims_dict["labels"][p][0][1] * yexpansion)

            tl_pred = [box[0][0] - tl_adj[0], box[0][1] - tl_adj[1]]

            tl_length = math.sqrt((tl_pred[0] - box[0][0]) ** 2 + (tl_pred[1] - box[0][1]) ** 2)
            tl_bc = abs(tl_pred[1] - box[0][1])

            ideal_angle = math.asin(tl_bc / tl_length)

            # however, need to account for angle, so, real coordinate will be

            adj_angle = ideal_angle - angle

            tl_pred_w_angle_x = box[0][0] - tl_length * math.cos(adj_angle)
            tl_pred_w_angle_y = box[0][1] - tl_length * math.sin(adj_angle) #testing: add y instead of minus since flipped

            angles.append(angle)
            exps_x.append(xexpansion)
            exps_y.append(yexpansion)
            form_tl_xs.append(tl_pred_w_angle_x)
            form_tl_ys.append(tl_pred_w_angle_y)

        #code.interact(local=locals())


        #noticing that the behavior of keras ocr is to favor locking boxes to 0, so
        #fine tuning this to be more responsive to any box angles.
        if angles.count(0) > len(angles)/2:
            angles_avg = sum(angles) / len(angles)
        else:
            #remove 0s and get median of reported angles.
            all_angles = [f for f in angles if f != 0]
            angles_avg = median(angles)

        angles_avg = sum(angles) / len(angles) #should probably not be avg. Perhaps avg if
        #mostly 0s, but otherwise median.
        exps_x_avg = sum(exps_x) / len(exps_x)
        exps_y_avg = sum(exps_x) / len(exps_x)
        form_tl_xs_avg = sum(form_tl_xs) / len(form_tl_xs)
        form_tl_ys_avg = sum(form_tl_ys) / len(form_tl_ys)

        return angles_avg,exps_x_avg,exps_y_avg,form_tl_xs_avg,form_tl_ys_avg

    def get_coords(self):

        self.predictions = self.predict()

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
                print(self.indeces_dict)
                #this assumes there's bin a hit. Locate the boxes, recreate the full title box, use to predict form
                #image2 = cv2.resize(self.image, (300, 400))
                #image2[round(749.1745244224945/10),round(2140.9488217836815/10)]=[0,0,255]
                # image2[round(2140.9488217836815/10),round(749.1745244224945/10)]=[0,0,255] #this works, exactly as is! Just need to invert when plotting
                #cv2.imshow('image', image2)
                #cv2.waitKey(0)

                #for each matched label in indeces_dict, run through a standard function and export
                angles_avg, exps_x_avg, exps_y_avg, form_tl_xs_avg, form_tl_ys_avg = self.getTrigMetrics()

                self.voucher_dims_dict['fields_adj'] = {}
                for m in self.voucher_dims_dict['fields']:

                    mod_field = get_mod_field(self.voucher_dims_dict['fields'][m],angles_avg,exps_x_avg, exps_y_avg, form_tl_xs_avg, form_tl_ys_avg)

                    #this adds to shallow copy of dict, does not modify original object
                    self.voucher_dims_dict['fields_adj'][m] = mod_field

                    #temporary, see how it is doing with angles in general
                    image2 = self.image.copy()
                    img_crop, _ = crop_rect(image2, self.voucher_dims_dict['fields_adj'][m])
                    cv2.imwrite(f"testout/{time.time()} cropped_img.jpg", img_crop)

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