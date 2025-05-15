import torch
from segment_anything import sam_model_registry, SamPredictor

def load_sam_model():
    sam_checkpoint = "models/sam/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor
