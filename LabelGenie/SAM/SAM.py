from .SegmentAnything.segment_anything import sam_model_registry, SamPredictor

class SAM:
    def __init__(self):
        sam = sam_model_registry["vit_b"](
            checkpoint="/home/jaydeep/dev/Reg/slideCV/SegmentAnything/checkpoint/sam_vit_b_01ec64.pth")
        self.predictor = SamPredictor(sam)
