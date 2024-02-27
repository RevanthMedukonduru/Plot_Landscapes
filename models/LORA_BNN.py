import open_clip
import os
from decouple import config
from bayes_wrap import generate_lora_model

import torch
from torch import nn
from bigmodelvis import Visualization

fix_start = True
num_bends = 3
fix_end = True
fix_points = [fix_start] + [False] * (num_bends - 2) + [fix_end]

# VIT-B-32 Model
mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

base_model = generate_lora_model(mdl, 1)
print("RECIEVED MODEL FINAL : ")
Visualization(base_model).structure_graph()

# Create a wrapper for the model    
class LORA_BNN:
    base = base_model
    preprocess = preprocess
    kwargs = {}