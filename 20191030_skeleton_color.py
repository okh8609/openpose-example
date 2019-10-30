#all together
import numpy as np
from matplotlib import pyplot as plt
import sys
import cv2
import os
from sys import platform
import argparse
sys.path.append('C:\\openpose\\build_CPU\\python\\openpose\\Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + 'C:\\openpose\\build_CPU\\x64\\Release;' + 'C:\\openpose\\build_CPU\\bin;'
import pyopenpose as op


inputImgDir = "C:\\NTUST_openpose\\img01\\"
outputImgDir = "C:\\NTUST_openpose\\img01_output\\"

files = os.listdir(inputImgDir)
for fileName in files:
    print(fileName)
    
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=inputImgDir+fileName, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
    # cv2.waitKey(0)

    # save to image
    img = cv2.imread(args[0].image_path)
    points = datum.poseKeypoints[0]
    # print(type(points))

    color = (200, 150, 0)
    a, b = 0, 1
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 1, 2
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 2, 3
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 3, 4
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 1, 8
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 1, 5
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 5, 6
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 6, 7
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 8, 9
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 9, 10
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 10, 11
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 8, 12
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 12, 13
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 13, 14
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 5, 6
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)
    a, b = 6, 7
    if (points[a][0],points[a][1])!=(0,0) and (points[b][0],points[b][1])!=(0,0) :
        cv2.line(img, (points[a][0],points[a][1]), (points[b][0],points[b][1]), color, 5)

    for cell in points:
    #     print(str(cell))
        if (cell[0], cell[1])!=(0,0) :
            cv2.circle(img,(cell[0], cell[1]), 7, (240, 80, 0), -1)

    cv2.putText(img, "Kai-Hao @ NTUST", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (75, 75, 75), 1, cv2.LINE_AA)

    cv2.imwrite(outputImgDir+fileName, img)

#     show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
#     plt.imshow(show_img)
#     plt.show()

#     cv2.imshow("NTUST", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
