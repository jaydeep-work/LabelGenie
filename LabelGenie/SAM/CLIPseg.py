from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from torch import nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CLIPseg:
    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    def txt_map(self, image, prompts):
        inputs = self.processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
        # predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        print(">>>>>>>>>>>> ", preds.shape)

        if len(preds.shape) == 3:
            preds = preds.reshape([1, 1, 352, 352])

        # resize the outputs
        preds = nn.functional.interpolate(
            preds,
            size=(image.size[1], image.size[0]),
            mode="bilinear"
        )
        return preds

    def img_map(self, image, prompt_img):
        encoded_image = self.processor(images=[image], return_tensors="pt")
        encoded_prompt = self.processor(images=[prompt_img], return_tensors="pt")
        # predict
        with torch.no_grad():
            outputs = self.model(**encoded_image, conditional_pixel_values=encoded_prompt.pixel_values)
        preds = outputs.logits.unsqueeze(1)
        preds = torch.transpose(preds, 0, 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(preds[0])
        plt.axis('on')
        plt.show()

        preds = preds.reshape([1, 1, 352, 352])
        # resize the outputs
        preds = nn.functional.interpolate(
            preds,
            size=(image.size[1], image.size[0]),
            mode="bilinear"
        )
        return preds

    def get_bbox_fro_txt(self, image, prompts, threshold):
        if type(prompts) == list:
            result = self.txt_map(image, prompts)
        else:
            result = self.img_map(image, prompts)
        bboxes = []
        points = []
        classes = []
        maps = []
        for i, map in enumerate(result):
            map = torch.sigmoid(map[0])
            map = np.where(map > threshold, map, 0)
            map = np.uint8(map * 255)
            maps.append(map)

            contours, hierarchy = cv2.findContours(map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                crop_tmp = map[y:y + h, x:x + w]
                y1, x1 = np.unravel_index(crop_tmp.argmax(), crop_tmp.shape)

                points.append([x1+x, y1+y])
                classes.append(i+1)
                bboxes.append([x, y, x+w, y+h])

            # print(x1, y1)
            # plt.imshow(crop_tmp)
            # plt.title("sdsfd")
            # plt.axis('off')
            # plt.show()

        return bboxes, classes, points, maps