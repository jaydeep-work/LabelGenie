import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

images = glob.glob('/home/jaydeep/dev/Reg/LabelGenie/eg_data/dogs/*.jpg')

for i in images:
    img = cv2.imread(i)
    print(img.shape)
    img_name = i.split("/")[-1].split(".")[-0]
    file1 = open(f"/home/jaydeep/dev/Reg/LabelGenie/eg_data/dogs/{img_name}.txt", "r+")
    # print(file1.readlines())

    for i in file1.readlines():
        cls, x, y, w, h = i.split(" ")
        cls = int(cls)
        x = float(x) * img.shape[1]
        y = float(y) * img.shape[0]
        w = float(w) * img.shape[1]
        h = float(h) * img.shape[0]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    plt.imshow(img)
    plt.title("sdsfd")
    plt.axis('off')
    plt.show()

