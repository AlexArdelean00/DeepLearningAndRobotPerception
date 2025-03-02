import os
import tarfile
import urllib.request
import csv
import glob
import xml.etree.ElementTree as ET
import ntpath
from matplotlib.pyplot import imread, imsave
import random

urlImages = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
urlAnnotations = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
dirnames = ['cat', 'dog']

def download_and_uncompress_tarball(tarball_url, dataset_dir):
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath)
    tarfile.open(filepath, "r:gz").extractall(dataset_dir)

def download_and_convert(data_root):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if not os.path.exists(os.path.join(data_root, "images")):
        print("[!] Downloading images...")
        download_and_uncompress_tarball(urlImages, data_root)
    
    if not os.path.exists(os.path.join(data_root, "annotations")):
        print("[!] Downloading annotations...")
        download_and_uncompress_tarball(urlAnnotations, data_root)

    data_dir = os.path.join(data_root, "pet")
    if os.path.exists(os.path.join(data_dir, "train.csv")) or \
       os.path.exists(os.path.join(data_dir, "test.csv")):
       return
    
    if not os.path.exists(os.path.join(data_dir, "train")):
        os.makedirs(os.path.join(data_dir, "train"))
    if not os.path.exists(os.path.join(data_dir, "test")):
        os.makedirs(os.path.join(data_dir, "test"))

    print("[!] Converting images...")
    # manage data with csv files
    train_f = open(os.path.join(data_dir, "train.csv"), "w")
    test_f  = open(os.path.join(data_dir, "test.csv"), "w")
    train_writer, test_writer = csv.writer(train_f, lineterminator='\n'), csv.writer(test_f, lineterminator='\n')

    imagePaths = glob.glob(os.path.join(data_root, "images", "*.jpg"))

    # use only images with bounding box
    imagePathsWithBB = []
    for path in imagePaths:
        annName = os.path.basename(path).rsplit('.', 1)[0] + ".xml"
        annPath = os.path.join(data_root, "annotations", "xmls", annName)
        if os.path.exists(annPath):
            imagePathsWithBB.append(path)

    # random.shuffle(imagePathsWithBB)
    # num_test = int(len(imagePathsWithBB)*0.2)
    # print(num_test)

    trainingImagePaths = glob.glob(os.path.join("../TrainBackbone/.data/pet/train/cat", "*.jpg"))\
          + glob.glob(os.path.join("../TrainBackbone/.data/pet/train/dog", "*.jpg"))
    testImagePaths = glob.glob(os.path.join("../TrainBackbone/.data/pet/test/cat", "*.jpg"))\
          + glob.glob(os.path.join("../TrainBackbone/.data/pet/test/dog", "*.jpg")) 

    # prepare a training data
    for path in imagePathsWithBB:
        head, tail = os.path.split(path)
        if any(tail in s for s in trainingImagePaths):
            annName = os.path.basename(path).rsplit('.', 1)[0] + ".xml"
            annPath = os.path.join(data_root, "annotations", "xmls", annName)

            tree = ET.parse(annPath)
            root = tree.getroot()
            filename = root.find('filename').text
            object_class = root.find('object/name').text
            xmin = int(root.find('object/bndbox/xmin').text)
            ymin = int(root.find('object/bndbox/ymin').text)
            xmax = int(root.find('object/bndbox/xmax').text)
            ymax = int(root.find('object/bndbox/ymax').text)

            if not os.path.exists(os.path.join(data_dir, "train", object_class)):
                os.makedirs(os.path.join(data_dir, "train", object_class))

            new_path = os.path.join(data_dir, "train", object_class, ntpath.split(path)[-1])
            im = imread(path)
            imsave(new_path, im)
            train_writer.writerow([new_path, object_class, xmin, ymin, xmax, ymax])

    # prepare a test data
    for path in imagePathsWithBB:
        head, tail = os.path.split(path)
        if any(tail in s for s in testImagePaths):
            annName = os.path.basename(path).rsplit('.', 1)[0] + ".xml"
            annPath = os.path.join(data_root, "annotations", "xmls", annName)

            tree = ET.parse(annPath)
            root = tree.getroot()
            filename = root.find('filename').text
            object_class = root.find('object/name').text
            xmin = int(root.find('object/bndbox/xmin').text)
            ymin = int(root.find('object/bndbox/ymin').text)
            xmax = int(root.find('object/bndbox/xmax').text)
            ymax = int(root.find('object/bndbox/ymax').text)

            if not os.path.exists(os.path.join(data_dir, "test", object_class)):
                os.makedirs(os.path.join(data_dir, "test", object_class))

            new_path = os.path.join(data_dir, "test", object_class, ntpath.split(path)[-1])
            im = imread(path)
            imsave(new_path, im)
            test_writer.writerow([new_path, object_class, xmin, ymin, xmax, ymax])

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou