import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import sys
from utility_functions import display_mask, get_boundry_img_matrix
import numpy as np, keras, cv2
from segmentation1 import segment_image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def pca(data):
    dvect = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j]: dvect.append([i, j])
    dvect = np.array(dvect, dtype=np.float32)
    dvect = np.array(dvect) - np.mean(dvect, axis=0)
    dvect /= np.std(dvect, 0)
    cov = np.dot(dvect.T, dvect) / dvect.shape[0]
    eigenval, eigenvect = np.linalg.eigh(cov)
    return cov, eigenvect, eigenval

if __name__ == "__main__":
    color = {i: np.random.randint(20, 255, 3) for i in range(5, 5000)}
    color[1] = [255, 255, 255]
    color[2] = [0, 0, 255]
    imgFile =  sys.argv[1]
    count = 1

    model = keras.models.load_model('weights/weights_01234567.pkl')
    
    #### Segmentation of grains ####
    segments, segLocation, _, mask= segment_image(imgFile)

    ##### Feature Extraction ####
    features = {}
    for gi in segments:
        gcolor = segments[gi]
        h, w, _ = gcolor.shape
        ggray = gcolor[:,:,2]
        thresh = np.array([[255 if pixel > 20 else 0 for pixel in row] for row in ggray])
        b = np.array(get_boundry_img_matrix(thresh, bval=1), dtype=np.float32)
        boundry = np.sum(b) / (h * w)
        area = np.sum(np.sum([[1.0 for j in range(w) if ggray[i, j]] for i in range(h)]))
        mean_area = area / (h * w)
        r, b, g = np.sum([gcolor[i, j] for j in range(w) for i in range(h)], axis=0) / (area * 256)
        _, _, eigen_value = pca(ggray)
        eccentricity = eigen_value[0] / eigen_value[1]
        l = [mean_area, boundry, r, b, g, eigen_value[0], eigen_value[1], eccentricity]
        features[gi] = np.array(l)

    out = {}
    #### Predicting the output ####
    for i in features:
        out[i] = model.predict(np.array([features[i]]))

    rect = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    good = not_good = 0
    for i in out:
        try:
            s = segLocation[i]
        except KeyError:
            continue
        if np.argmax(out[i][0]) == 0:
            good += 1
            rect = cv2.rectangle(rect, (s[2], s[0]), (s[3], s[1]), (0, 0, 0), 1)
        else:
            not_good+=1
            rect = cv2.rectangle(rect, (s[2], s[0]), (s[3], s[1]), (0, 0, 255), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = rect.shape
    cv2.putText(rect, text='Number of good grain: %d  Number Not good grain or impurity: %d'%(good,not_good), org=(10, h - 50), fontScale=1, fontFace=font, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    maskFile = 'mask_'+imgFile.split('/')[-1]
    outFile = 'result_'+imgFile.split('/')[-1]
    cv2.imwrite(outFile, rect)
    display_mask('mask',mask,sname=maskFile)
    cv2.waitKey(0)
    count+=1

def predict(path):
    color = {i: np.random.randint(20, 255, 3) for i in range(5, 5000)}
    color[1] = [255, 255, 255]
    color[2] = [0, 0, 255]
    imgFile =  path
    count = 1

    model = keras.models.load_model('weights/weights_01234567.pkl')
    
    #### Segmentation of grains ####
    segments, segLocation, _, mask= segment_image(imgFile)

    ##### Feature Extraction ####
    features = {}
    for gi in segments:
        gcolor = segments[gi]
        h, w, _ = gcolor.shape
        ggray = gcolor[:,:,2]
        thresh = np.array([[255 if pixel > 20 else 0 for pixel in row] for row in ggray])
        b = np.array(get_boundry_img_matrix(thresh, bval=1), dtype=np.float32)
        boundry = np.sum(b) / (h * w)
        area = np.sum(np.sum([[1.0 for j in range(w) if ggray[i, j]] for i in range(h)]))
        mean_area = area / (h * w)
        r, b, g = np.sum([gcolor[i, j] for j in range(w) for i in range(h)], axis=0) / (area * 256)
        _, _, eigen_value = pca(ggray)
        eccentricity = eigen_value[0] / eigen_value[1]
        l = [mean_area, boundry, r, b, g, eigen_value[0], eigen_value[1], eccentricity]
        features[gi] = np.array(l)

    out = {}
    #### Predicting the output ####
    for i in features:
        out[i] = model.predict(np.array([features[i]]))

    rect = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    good = not_good = 0
    for i in out:
        try:
            s = segLocation[i]
        except KeyError:
            continue
        if np.argmax(out[i][0]) == 0:
            good += 1
            rect = cv2.rectangle(rect, (s[2], s[0]), (s[3], s[1]), (0, 0, 0), 1)
        else:
            not_good+=1
            rect = cv2.rectangle(rect, (s[2], s[0]), (s[3], s[1]), (0, 0, 255), 3)

    return good, not_good