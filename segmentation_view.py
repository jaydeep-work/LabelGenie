import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

images = glob.glob('/home/jaydeep/dev/Reg/LabelGenie/eg_data/humans/*.png')

for i in images:
    img = cv2.imread(i)
    print(img.shape, np.unique(img))

    for j in np.unique(img)[1:]:
        img_ = np.where(img == j, 255, 0)
        # img_ = np.where(img != 0, 255, 0)
        plt.imshow(img_)
        plt.title(f"{i.split('/')[-1]}-{j}")
        plt.axis('off')
        plt.show()
