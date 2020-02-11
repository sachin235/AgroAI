import cv2, numpy as np
# from Area import areaThreshold_by_havg, areaThreshold_by_top
# from threshold import otsu_threshold
# from _8connected import get_8connected_v2
from util import *
import warnings
import traceback
from eclipes_test import elliptic_fourier_descriptors, efd
warnings.filterwarnings("error")
import time
#print imshow
color = {i: np.random.randint(20, 255, 3) for i in range(5, 5000)}
color[1] = [255, 255, 255]
color[2] = [0, 0, 255]
def areaThreshold_by_avg(axis, exp):
    avga = np.average([(s[1] - s[0]) * (s[3] - s[2]) for i,s in axis.items()])
    low = avga/2**exp
    high = avga*2**exp
    return low, high
def get_equiv(equivlence, labels, mcount):
    for i in reversed(sorted(list(equivlence))):
        for j in equivlence[i]:
            if j in equivlence:
                equivlence[i] = list(set(equivlence[i] + equivlence[j]))
    a = sorted(list(equivlence))
    for i in reversed(range(len(a))):
        for j in equivlence[a[i]]:
            if j in equivlence:
                equivlence.pop(j)
            if j in labels:
                labels.pop(labels.index(j))

    out_labels = {}
    for i in labels:
        out_labels[i] = mcount
        mcount+=1

    for i in equivlence:
        for j in equivlence[i]:
            out_labels[j] = out_labels[i]
    return out_labels
def otsu_threshold(gray):
    h,w = gray.shape
    count = {i:0 for i in range(256)}
    for i in range(h):
        for j in range(w):
            count[gray[i,j]] += 1
    prob = np.array([count[i]/float(h*w) for i in sorted(count)])
    means = np.array([prob[i]*(i+1) for i in count])
    mean = np.sum(means)
    minvar = -np.inf
    minT = 0
    for t in range(256):
        w1 = np.sum([i for i in prob[:t+1]])
        w2 = 1.0-w1
        if not w1 or not w2: continue
        m1 = np.sum([i for i in means[:t+1]])
        mean1 = m1/w1
        mean2 = (mean - m1)/w2
        bcvar = w1*w2*(mean2-mean1)**2
        if bcvar > minvar:
            minvar = bcvar
            minT = t
    return minT
    
def areaThreshold_by_havg(axis, exp):
    areas = np.sort([(s[1] - s[0]) * (s[3] - s[2]) for i,s in axis.items()])
    alen = len(areas)
    avga = np.average([areas[i] for i in range(int(alen/2**exp), int(alen*(1-1.0/2**exp)))])
    low = avga / 2 ** exp
    high = avga * 2 ** exp
    return low, high

def areaThreshold_by_top(axis, exp):
    area = np.sort([(s[1] - s[0]) * (s[3] - s[2]) for i, s in axis.items()])[-1]
    return area/2**exp , area*2**exp

def get_8connected_v2(thresh, mcount=5):   # _8connected
    h,w=thresh.shape
    image_label = np.zeros((h,w), dtype=np.int)
    label = 1
    equivlence = {}
    kernal = np.array([
                        [1,1,1],
                        [1,1,1],
                        [1,1,1]
                      ], dtype=np.uint8)

    image_label = padding2D_zero(image_label,1)
    thresh = padding2D_zero(thresh,1)
    out_labels = []
    for i in range(1,h+1):
        for j in range(1,w+1):
            if thresh[i,j] == 255 and image_label[i,j] == 0:
                labels = set(image_label[i-1:i+2,j-1:j+2].reshape(3*3).tolist())
                labels = sorted(labels)
                labels.pop(0) if labels[0] == 0 else None
                if labels:
                    val = labels[0]
                else:
                    label += 1
                    val = label
                    out_labels.append(label)
                image_label[i-1:i+2,j-1:j+2] = (thresh[i-1:i+2,j-1:j+2]/255)*val
                if len(labels) > 1:
                    if labels[0] in equivlence:
                        equivlence[labels[0]] = list(set(equivlence[labels[0]] +labels[1:]))
                    else:
                        equivlence[labels[0]] = labels[1:]
    image_label = remove_padding2D_zero(image_label,1)
    # ##print equivlence
    # ##print out_labels
    # display_mask('image label',image_label)
    # raw_input()
    seg = get_equiv(equivlence, out_labels, mcount)

    for i in range(h):
        for j in range(w):
            if image_label[i, j]:
                image_label[i, j] = seg[image_label[i, j]]
    # ##print seg
    return image_label

def make_border(points, shape, bval=255):
    # h,w = shape
    boundry = np.zeros(shape, dtype=np.uint8)
    # boundry = padding2D_zero(boundry,2)
    boundry[points[0][0],points[0][1]] = bval
    i=0
    x,y = points[0]
    while i < len(points)-1:
        try:
            boundry[x, y] = bval
        except IndexError:
            x1=int(x);y1=int(y)
            if x >= boundry.shape[0]:
                x1 = int(boundry.shape[0])-1
            if y >= boundry.shape[1]:
                y1 = int(boundry.shape[1])-1
            boundry[x1, y1] = bval
            # traceback.#print_exc()
        if abs(points[i+1][0] - x) <=1 and abs(points[i+1][1] - y) <=1:
            i+=1
            x,y = points[i]
        elif abs(points[i+1][0] - x) > 1:
            x ,y = (x + (points[i+1][0] - x)/abs(points[i+1][0] - x)), y
        elif abs(points[i+1][1] - y) > 1:
            x ,y = x, (y + (points[i+1][1] - y)/abs(points[i+1][1] - y))
    # boundry = remove_padding2D_zero(boundry, 2)
    return boundry

def mask_by_border(boundry, ival):
    h,w = boundry.shape
    inside = 0
    b1=np.int0(boundry)
    for i in range(h):
        # try:
        val = np.ones(np.argmax(b1[i,:])) * 2
        b1[i,:len(val)] = val
        val1 = np.ones(np.argmax(b1[i,::-1])) *2
        b1[i,w-len(val1):] = val1
    for i in range(w):
        val = np.ones(np.argmax(b1[::-1,i])) * 2
        b1[h-len(val):,i] = val
        val = np.ones(np.argmax(b1[:,i])) * 2
        b1[:len(val),i] = val
    b1 = ((b1 - boundry)/-2 + 1) * ival
    return b1

def L2_segmentation_2(iimg , T, index):
    h, w, _ = iimg.shape
    # cv2.imshow('image', iimg)
    t0 = time.time()
    gray = iimg[:, :, 2]
    # cv2.imshow('gray', gray)

    thresh = np.array([[0 if pixel < T else 255 for pixel in row] for row in gray], dtype=np.uint8)


    sober = sober_operation(gray)
    # cv2.imshow('sober', sober)
    # #print "\tsober operation", time.time() - t0
    # t0 = time.time()

    sober = cv2.fastNlMeansDenoising(sober, None, h=2, templateWindowSize=3, searchWindowSize=5)
    # cv2.imshow('sober cleaned', sober)
    # #print "\tnoise operation", time.time() - t0
    # t0 = time.time()

    T= otsu_threshold(sober)
    # #print "\tsober threshold", time.time() - t0
    # t0 = time.time()

    sthresh = np.array([[0 if pixel < T else 255 for pixel in row] for row in sober], dtype=np.uint8)
    # cv2.imshow('sober Threshold', sthresh)
    # #print "\tcalc threshold", time.time() - t0
    # t0 = time.time()

    diluted = cv2.dilate(sthresh, kernel=np.ones((5,5), np.uint8), iterations=1)
    # cv2.imshow('dilutated2 ', diluted)
    # #print "\tdilation operation", time.time() - t0
    # t0 = time.time()

    thresh2 = np.where((thresh == 0) * (diluted == 255), 0, thresh-diluted)
    # cv2.imshow('Thresh - dilute ', thresh2)

    mask = get_8connected_v2(thresh=thresh2, mcount=index)
    # display_mask("Diluted mask", mask)
    # #print "\tmask foamation", time.time() - t0
    # t0 = time.time()

    # Calcutaing the grain segment using mask image
    s = cal_segment_area(mask)
    # #print "\tcalc area seg", time.time() - t0
    # t0 = time.time()

    rmcount = 0
    if len(s) < 2:
        # #print
        return None
    low_Tarea, up_Tarea = areaThreshold_by_top(s, 3)
    slist = list(s)
    for i in slist:
        area = (s[i][0] - s[i][1]) * (s[i][2] - s[i][3])
        if area < low_Tarea:
            s.pop(i)
            rmcount += 1
    if len(s) < 2:
        # #print
        return None
    # #print "\tselecting area", time.time() - t0
    # t0 = time.time()

    # removing unwanted masks
    mask = np.array([[0 if pixel not in s else pixel for pixel in row] for row in mask])
    # #print "\tremoving unwanted mask opeation", time.time() - t0
    # t0 = time.time()

    # Adding boundry mask
    boundry = get_boundry_img_matrix(thresh, 1)
    # #print "\tgetting boundry", time.time() - t0
    # t0 = time.time()

    mask = np.where(boundry == 1, 1, mask)
    # #print "\tadding boundry to mask opeation", time.time() - t0
    # t0 = time.time()

    # display_mask('boundried mask', mask)

    # using mask fill the mask values in boundry
    mask = flood_filling(mask)
    # #print "\tflood filling opeation", time.time() - t0
    # t0 = time.time()
    # display_mask('flood fill', mask)

    # replace boundry by respective mask value
    mask = boundry_fill(mask)
    # #print "\tfilling opeation", time.time() - t0
    # t0 = time.time()
    # cv2.waitKey()

    masks =[]
    for ii in s:
        img = get_mask_value_area(gray, mask, ii)
        # b1 = get_boundry_img_matrix(img)
        # b2 = get_boundry_img_matrix(get_mask_value_area(boundry, mask, i),bval=255)
        # img = b1-b2
        points = get_boundry_as_points(img)
        img = get_boundry_img_matrix(img, bval=255)
        # cv2.imshow("img %d" % (ii), img)
        coff = elliptic_fourier_descriptors(points,order=5)
        if coff is None:
            #print "Ellipsis not work"
            return None
        x, y = np.int0(efd(coff, contour_1=points, locus=np.mean(points, axis=0)))
        tt=list(zip(x,y))
        boundry = make_border(tt, img.shape, bval=255)
        # cv2.imshow("border %d"%(ii), boundry)
        mask1 = mask_by_border(boundry, ii)
        # display_mask("mask %d" % (ii), mask1)
        masks.append(mask1)
    # #print "\telliptical fitting operation", time.time() - t0,'\n'

    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return masks, rmcount
