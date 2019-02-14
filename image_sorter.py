# !/anaconda3/envs/dlib python3.6

import os, cv2, imutils, shutil,sys
import numpy as np
from imutils.paths import list_images

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
IMG_VAR = 160
IMG_HIST_DISTANCE = 0.8

import face_recognition, pathlib
from face_recognition import face_distance
detect_method = ["hog", "cnn"]

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

################################################
################################################
# input a face image path; return a face_encoding list that represent this face image
# 输入一张人脸box图像,返回改人脸box的编码列表;如果是一张图片,可能有几张人脸,则返回的列表包含这几张人脸
def encode_face(face_image):
    face_img = cv2.imread(face_image)
    face_RGB = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # face_RGB = skimage.io.imread(face_image)
    # print(detect_method[1])
    boxes = face_recognition.face_locations(face_RGB, model=detect_method[0])
    encodings = face_recognition.face_encodings(face_RGB,boxes)
    known_encoding = []
    # known_name = face_name
    for encoding in encodings:
        known_encoding.append(encoding)
    return known_encoding
    # return a list

def image_file_list(folder):
    img_list = []
    for (root, dirs, files) in os.walk(folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                img_list.append(file)
    return img_list

## 搜索相似的人脸,阈值0.4, 可调
# 输入: 一张人脸box图片, 和待搜索的包含人脸box图片的目录; 输出: 字典{每张图片路径:[这张图片中是否和源人脸相同的真/假值的列表]}
# 调用了encode_face()将人脸图像进行编码
def search_face(face_image_src, face_img_dir_to_test):

    face_src_encoding_list = encode_face(face_image_src)

    results = {}
    for (root, dirs, files) in os.walk(face_img_dir_to_test):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                file_path = os.path.join(os.getcwd(), face_img_dir_to_test, file)
                for i in face_src_encoding_list:
                    match = face_recognition.compare_faces(encode_face(file_path), i)
                    print("\t match in search_face: ", match)
                    results[file_path] = match

    return results

##########
## 输入: search_same_face()函数返回的目录中所有人脸图像的字典
## 输出: 相同的人脸的图像的完整路径列表
def get_same_faces(results):

    same_faces = []
    for img_path in results.keys():
        for t in results[img_path]:
            if t: same_faces.append(img_path)
        # faces = [img_path for t in results[img_path] if t]
    return same_faces

def get_outlier_faces(results):

    outlier_faces = []
    for img_path in results.keys():

        # for t in results[img_path]:
        #     print("Get outlier: ", t)
        #     if not t: outlier_faces.append(img_path)
        # # faces = [img_path for t in results[img_path] if t]
        ### 默认每张图片上只有一个人脸, 如果 dlib没有检测出人脸, 列表的值为空 #####
        if results[img_path]==[]:
            outlier_faces.append(img_path)
    print("outlier_faces : ",outlier_faces)
    return outlier_faces
##########

################################################
################################################

def main(src_img, samples_folder, test_images_folder):
    indexes = index_images(samples_folder)

    # make a dir to store outer faces
    not_face = "not_target_face"
    if not os.path.exists(not_face):
        os.mkdir(not_face)

    for image in list_images(test_images_folder):
        if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg"):
            # 颜色直方图对比
            score = compare_new_image(indexes, image)
            # 图像二阶方差对比
            # v = image_variance(image)
            # print("{} variance={:.0f}{sign}\n".format(image, v ,sign = " < {} :to be removed !".format(IMG_VAR) if v<IMG_VAR else "."))

            # 这里可以调整这个图片方差值对比值来删除不合乎要求的图片
            # if min(score) >= IMG_HIST_DISTANCE or image_variance(image) < IMG_VAR:
            #     shutil.move(image, os.path.join(CURRENT_PATH, not_face))

            if min(score) >= IMG_HIST_DISTANCE:
                shutil.move(image, os.path.join(CURRENT_PATH, not_face))

    # print("\t Start search faces ... ")
    results = search_face(src_img, test_images_folder)
    print(results)
    outlier_faces = get_outlier_faces(results)
    print(outlier_faces)
    for image in outlier_faces:
        # print("outlier image: ", image)
        shutil.move(image, os.path.join(CURRENT_PATH, not_face))


if __name__ == "__main__":

    # # 获取不同人脸的图像路径列表
    # r = search_face("sample/00401_0.png", "sample")
    # print(get_outlier_faces(r))

    # src_face = sys.argv[1]
    # samples_folder = sys.argv[2]
    # test_images_folder = sys.argv[3]

    src_face = input("输入一张人脸对比筛选用的人脸参考图片(路径): ")
    samples_folder = input("输入颜色直方图筛选的源图片目录: ")
    test_images_folder = input("输入要测试的人脸图片目录: ")

    main(src_face, samples_folder, test_images_folder)