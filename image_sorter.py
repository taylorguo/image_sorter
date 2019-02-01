# !/anaconda3/envs/dlib python3.6

import os, cv2, imutils, shutil,sys
import numpy as np
from face_recognition import face_distance
from imutils.paths import list_images

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
IMG_VAR = 190
IMG_HIST_DISTANCE = 0.8

class Histogram3D:
    def __init__(self, bins):
        self.bins = bins
    def describe(self, image):
        hist = cv2.calcHist([image], [0,1,2], None, self.bins, [0,256,0,256,0,256])
        if imutils.is_cv2(): hist = cv2.normalize(hist)
        else: hist= cv2.normalize(hist, hist)
        return hist.flatten()

class SearchEngine:
    def __init__(self, index):
        self.index = index
    def chi2distance(self, hA, hB, eps = 1e-10):
        d = 0.5 * np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(hA,hB)])
        return d
    def search(self, nfeatures):
        results = {}
        for (k, features) in self.index.items():
            results[k] = self.chi2distance(features, nfeatures)
        results = sorted([(v, k) for (k, v) in results.items()])
        return results

### input a images folder, output a dict - {image name: hist vector}
def index_images(dataset_path):
    index = {}
    descriptor = Histogram3D([8, 8, 8])
    for image_path in list_images(dataset_path):
        image = image_path[image_path.rfind("/") + 1:]
        index[image] = descriptor.describe(cv2.imread(image_path))
    print("[INFO] done...indexed {} images".format(len(index)))
    return index

## 搜索相似的人脸,阈值0.4, 可调
def search_faces(known_face_encodings, face_encoding_to_check, tolerance=0.4):
    # print(face_distance(known_face_encodings, face_encoding_to_check))
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def average_score(scores_list):
    nsum = 0
    for i in scores_list: nsum += i
    average = nsum / (len(scores_list))
    return average

def compare_new_image(sample_indexes, testimage):
    test_image = cv2.imread(testimage)
    # cv2.imshow("Searching", test_image)

    descriptor = Histogram3D([8,8,8])
    nfeatures = descriptor.describe(test_image)

    searcher = SearchEngine(sample_indexes)
    results = searcher.search(nfeatures)

    scores = []
    for i in range(0, len(sample_indexes)):
        (score, image_name) = results[i]
        scores.append(score)

    nsum = 0
    for i in scores: nsum += i
    average = nsum / (len(scores))
    print("Sample_imgs-{} = {:.2f}".format(testimage,average))

    # print(scores)
    return scores

def image_variance(image):
    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return img_variance

def main(samples_folder, test_images_folder):
    indexes = index_images(samples_folder)

    # make a dir to store outer faces
    not_face = "not_target_face"
    if not os.path.exists(not_face):
        os.mkdir(not_face)

    for image in list_images(test_images_folder):
        if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg"):
            score = compare_new_image(indexes, image)
            v = image_variance(image)
            print("{} variance={:.0f}{sign}\n".format(image, v ,sign = " < 190 :to be removed !" if v<IMG_VAR else "."))
            # 这里可以调整这个对比值来删除不合乎要求的图片
            if min(score) >= IMG_HIST_DISTANCE or image_variance(image) < IMG_VAR:
                shutil.move(image, os.path.join(CURRENT_PATH, not_face))

if __name__ == "__main__":
    samples_folder = sys.argv[1]
    test_images_folder = sys.argv[2]
    main(samples_folder, test_images_folder)
