from LabelGenie import LabelGenie

engine = LabelGenie(model='SAM')

# engine.segmentation(inp_path='/home/jaydeep/dev/Reg/LabelGenie/eg_data/dogs/',
#                     out_path='/home/jaydeep/dev/Reg/LabelGenie/eg_data/dogs/', prompt=[["puppies", "puppy's", "dog"], 'person'], threshold=0.3)

# engine.segmentation(inp_path='/home/jaydeep/dev/Reg/LabelGenie/eg_data/humans/',
#                     out_path='/home/jaydeep/dev/Reg/LabelGenie/eg_data/humans/',
#                     prompt='/home/jaydeep/dev/Reg/LabelGenie/eg_data/humans/refrance_img.jpeg', threshold=0.3)


engine.obj_detection(inp_path='/home/jaydeep/dev/Reg/LabelGenie/eg_data/dogs/',
                    out_path='/home/jaydeep/dev/Reg/LabelGenie/eg_data/dogs/', prompt=[["puppies", "puppy's", "dog"], 'person'],
                    threshold=0.3, bbox_type='yolo')
# engine.obj_detection(inp_path='', out_path='', prompt=[], threshold=0.6, bbox_type='yolo')


# TODO
# CLIP alternatives
    # https://huggingface.co/docs/transformers/model_doc/owlvit

# segmentation - connected contours
        # img to img...
        # multi prompt in img to img
        # sub-loop masking (Done)
# Object detection
        # masks to BBOX
# mask and bbox viewer


# Limitations
    # only sematic segmentation can be done not instance segmentation.


# Notes
# CLIP-Interrogator (for finding relevant prompt names.)
        # https://replicate.com/pharmapsychotic/clip-interrogator


