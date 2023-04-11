from PIL import Image
import numpy as np

def process_png(png):
    png = png.convert('RGBA')
    background = Image.new('RGBA', png.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, png)

    alpha_composite_3 = alpha_composite.convert('RGB')
    return alpha_composite_3

def get_flat_prompt_map(prompts):
    class_map = {}
    cls_counts = 1
    flat_prompt = []
    for i, cls in enumerate(prompts):
        if type(cls) == str:
            class_map[cls_counts] = i + 1
            cls_counts += 1
            flat_prompt.append(cls)
        else:
            flat_prompt += cls
            for j in cls:
                class_map[cls_counts] = i + 1
                cls_counts += 1
    return flat_prompt, class_map



def NMS(boxes, overlapThresh = 0.4):
    boxes = np.array(boxes)
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    #return only the boxes at the remaining indices
    return boxes[indices].astype(int)


def save_txt(image_name, path, bbox, width, height):
    txt = []
    for i in bbox:
        txt.append(f"{i[-1]} {(i[0]/width).round(3)} {(i[1]/height).round(3)} {(i[2]/width).round(3)} {(i[3]/height).round(3)}\n")

    file1 = open(f"{path}/{image_name.split('.')[0] + '.txt'}", "w")
    file1.writelines(txt)
    file1.close()
    print(f"{path}/{image_name.split('.')[0] + '.txt'}")