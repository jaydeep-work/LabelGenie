from .SAM import Obj_Detection, Segmentation
from .SAM.SegmentAnything.segment_anything import sam_model_registry, SamPredictor

class LabelGenie:
    def __init__(self, model='SAM'):
        self.supported_models = ['SAM']
        if model not in self.supported_models:
            assert f"'{model}' is not supported. currently we support only {self.supported_models}"

        self.model = model
        self.segmentation_logic = Segmentation()
        self.obj_detection_logic = Obj_Detection()

    def obj_detection(self, inp_path, out_path, prompt, threshold=0.6, bbox_type='yolo'):
        if self.model == "SAM":
            self.obj_detection_logic.run(inp_path, out_path, prompt, threshold=threshold, bbox_type=bbox_type)


    def segmentation(self, inp_path, out_path, prompt, threshold=0.6):
        if self.model == "SAM":
            self.segmentation_logic.run(inp_path, out_path, prompt, threshold=threshold)

