import glob
from functools import reduce
import tqdm
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from .SAM import SAM
from .CLIPseg import CLIPseg
from .sam_utils import process_png, get_flat_prompt_map, NMS, save_txt
import cv2

class Obj_Detection:
    def __init__(self):
        self.supported_image_extensions = ['.jpg', '.png', '.jpeg']
        self.predictor = SAM().predictor
        self.CLIP_seg = CLIPseg()

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        return mask_image

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def masks_from_bbox(self, img, bboxs, clses, class_map, image_name, out_path):
        y, x = img.size
        input_boxes = torch.tensor(bboxs, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, (x, y))
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False, )

        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(img)
        final_mask = np.zeros((img.size[1], img.size[0]))
        for i, mask in enumerate(masks):
            tmp_mask = mask.cpu().numpy()[0]
            final_mask += np.where(tmp_mask == True, np.where(final_mask == 0, class_map[clses[i]], 0), 0)
            # print(i, clses[i], mask.cpu().numpy().shape, np.unique(final_mask))

        # final_mask = Image.fromarray(final_mask).convert("L")
        # final_mask.save(f"{out_path}/{image_name.split('.')[0] + '.png'}")

        bboxs_final = []
        classes_final = []
        print(np.unique(final_mask))
        for j in np.unique(final_mask)[1:]:
            final_mask_ = np.where(final_mask == j, 255, 0)
            final_mask_ = np.uint8(final_mask_)

            kernel = np.ones((5, 5), np.uint8)
            final_mask_ = cv2.dilate(final_mask_, kernel, iterations=1)
            area_threshold = 1000

            contours, hierarchy = cv2.findContours(final_mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                if area >= area_threshold:
                    bboxs_final.append([x, y, w, h, j])
                    # print([x, y, w, h], area)

            # plt.imshow(final_mask_)
            # plt.title(f"{j}")
            # plt.axis('off')
            # plt.show()

        bboxs_final = NMS(bboxs_final, 0.7)
        print(bboxs_final)
        print(final_mask.shape)
        save_txt(image_name, out_path, bboxs_final, final_mask.shape[1], final_mask.shape[0])

        return masks

    def run(self, inp_path, out_path, prompt, threshold=0.6, bbox_type='yolo'):
        total_images = [glob.glob(inp_path + f"/*{i}") for i in self.supported_image_extensions]
        total_images = reduce(lambda z, y: z + y, total_images)
        for i in tqdm.tqdm(total_images):
            # read image
            try:
                img = Image.open(i)
                if np.asarray(img).shape[-1] == 4:
                    img = process_png(img)
            except Exception as e:
                raise Exception(f"Error in reading image '{i}' : {e}")

            # reading prompt image
            try:
                if type(prompt) == str:
                    prompt_img = Image.open(prompt)
                    if np.asarray(prompt_img).shape[-1] == 4:
                        prompt_img = process_png(prompt_img)
            except Exception as e:
                raise Exception(f"Error in reading prompt image '{prompt}' : {e}")

            # extracting ROI bounding boxes using CLIP
            try:
                print(prompt, type(prompt))
                if type(prompt) == str:
                    bboxs, clses, points, maps = self.CLIP_seg.get_bbox_fro_txt(img, prompt_img, threshold)
                    class_map = {1: 1}
                else:
                    flat_prompt, class_map = get_flat_prompt_map(prompt)
                    bboxs, clses, points, maps = self.CLIP_seg.get_bbox_fro_txt(img, flat_prompt, threshold)
            except Exception as e:
                raise Exception(f"Error in CLIPseg '{i}' : {e}")

            # creating image embedding
            try:
                self.predictor.set_image(np.array(img))
            except Exception as e:
                raise Exception(f"Error in creating image embedding SAM '{i}' : {e}")

            # extracting mask
            # try:
            image_name = i.split("/")[-1]
            self.masks_from_bbox(img, bboxs, clses, class_map, image_name, out_path)
            # except Exception as e:
            #     raise Exception(f"Error in mask extraction from SMA-CLIP '{i}' : {e}")
