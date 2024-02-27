

"""Train CIFAR10 with PyTorch."""
import copy
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import grad
from decouple import config
from src.modeling import ImageClassifier, ImageEncoder
from src.heads import get_classification_head

from opendelta import LoraModel, AdapterModel, LowRankAdapterModel # use lora as an example, others are same

import re
import importlib
from bigmodelvis import Visualization
from opendelta.utils.inspect import inspect_module_statistics

seed = 113
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class SVGD(nn.Module):
    def __init__(self, NET, opt):
        super().__init__()

        num_particles = int(config('opt'))
        self.h_kernel = 0
        self.particles = []

        for i in range(num_particles):
            self.particles.append(copy.deepcopy(NET))
            
        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)

        # logging.info("num particles: %d" % len(self.particles))
        print(f"num particles: {len(self.particles)}")

    def sample_particle(self):
        return self.particles[np.random.randint(0, len(self.particles))]

    def get_particle(self, index):
        return self.particles[index]

    def forward(self, x, **kwargs):
        logits, entropies, soft_out, stds = [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        for particle in self.particles:
            l = particle(x)
            sft = torch.softmax(l, 1)
            soft_out.append(sft)
            logits.append(l)
            if return_entropy:
                l = torch.softmax(l, 1)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
        logits = torch.stack(logits).mean(0)
        soft_out = torch.stack(soft_out).mean(0)
        return logits

    def update_grads(self):
        if np.random.rand() < 1 - float(config('p_update')):
            return
        all_pgs = self.particles
        if self.h_kernel <= 0:
            self.h_kernel = 0.1  # 1
        dists = []
        alpha = float(config('alpha'))  # if t < 100 else 0.0
        new_parameters = [None] * len(all_pgs)

        for i in range(len(all_pgs)):
            new_parameters[i] = {}
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is None:
                    new_parameters[i][l] = None
                else:
                    new_parameters[i][l] = p.grad.data.new(
                        p.grad.data.size()).zero_()
            for j in range(len(all_pgs)):
                # if i == j:
                #     continue
                for l, params in enumerate(
                        zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                    p, p2 = params
                    if p.grad is None or p2.grad is None:
                        continue
                    if p is p2:
                        dists.append(0)
                        new_parameters[i][l] = new_parameters[i][l] + \
                            p.grad.data
                    else:
                        d = (p.data - p2.data).norm(2)
                        # if p is not p2:
                        dists.append(d.cpu().item())
                        kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
                        new_parameters[i][l] = (
                            ((new_parameters[i][l] + p2.grad.data) -
                             (d / self.h_kernel**2) * alpha) /
                            float(len(all_pgs))) * kij
        self.h_kernel = np.median(dists)
        self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)) + 1)
        for i in range(len(all_pgs)):
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is not None:
                    p.grad.data = new_parameters[i][l]

    def get_losses(self, image, labels, criterion, **kwargs):
        losses = []
        for particle in self.particles:
            l = particle(image)
            loss = criterion(l, labels)
            losses.append(loss)
        losses = torch.stack(losses).mean(0)
        # if torch.abs(losses - loss) > 1e-5:
        #     print("Losses are not equal")
        return losses

class LP1WithAdapter(nn.Module):
    def __init__(self, model, img):
        super().__init__()
        device = f"cuda:{str(config('device'))}" if torch.cuda.is_available() else "cpu"

        model_module = importlib.import_module("model")
        count_trainable_parameters_lora = model_module.count_trainable_parameters

        # self.model = model # this is the VIT-B model
        # self.clip_model = model.image_encoder.model.visual.to(device) # this is the backbone model
        self.clip_model = model.to(device) # this is the backbone model

        print("\nEntered LP1WITHADAPTER:")
        Visualization(self.clip_model).structure_graph()
        trainable, delta, total = count_trainable_parameters_lora(self.clip_model, None)
        print(f"[VIT-B/ENTERED MODEL] Trainable Params: {trainable} | Delta Params: {delta} | Total Params: {total}")


        print("\nActual Backbone Model:")
        Visualization(self.clip_model).structure_graph()
        trainable, delta, total = count_trainable_parameters_lora(self.clip_model, None)
        print(f"[BACKBONE MODEL] Trainable Params: {trainable} | Delta Params: {delta} | Total Params: {total}")
        # dummy_inputs = torch.randn(1, 3, 32, 32).cuda()
        # dummy_inputs = dummy_inputs.to(torch.float) 
        dummy_inputs = img.long()
        dummy_inputs = dummy_inputs.to(device)
        setattr(self.clip_model, "dummy_inputs", dummy_inputs)
        # FIXME it runs out of CUDA here
        delta_model_temp = AdapterModel(backbone_model=self.clip_model, 
                                # modified_modules=['c_fc', 'c_proj'],
                                modified_modules=['transformer.resblocks.0.mlp.c_proj'],
                                bottleneck_dim=4,
                                #                     'transformer.resblocks.1.mlp.c_proj', 
                                #                     'transformer.resblocks.2.mlp.c_proj', 
                                #                     'transformer.resblocks.3.mlp.c_proj', 
                                #                     'transformer.resblocks.4.mlp.c_proj', 
                                #                     'transformer.resblocks.5.mlp.c_proj', 
                                #                     'transformer.resblocks.6.mlp.c_proj', 
                                #                     'transformer.resblocks.7.mlp.c_proj', 
                                #                     'transformer.resblocks.8.mlp.c_proj', 
                                #                     'transformer.resblocks.9.mlp.c_proj',
                                #                     'transformer.resblocks.10.mlp.c_proj', 
                                #                     'transformer.resblocks.11.mlp.c_proj'],
                                )
        self.delta_model = delta_model_temp

        print("\nBackbone Model after adding LoRa:")
        self.delta_model.log()
        trainable_params, delta_params, total_params = count_trainable_parameters_lora(delta_model_temp.backbone_model, delta_model_temp.modified_modules)
        print(f"[BACKBONE MODEL & LoRa] Trainable Params: {trainable_params} | Delta Params: {delta_params} | Total Params: {total_params}")

        print("\nBackbone Model after adding LoRa, and FROZEN:")
        self.delta_model.freeze_module(exclude=["deltas", "ln_final"], set_state_dict=True)
        self.delta_model.log()
        trainable_params, delta_params, total_params  = count_trainable_parameters_lora(delta_model_temp.backbone_model, delta_model_temp.modified_modules)
        print(f"[BACKBONE MODEL & LoRa & FROZEN] Trainable Params: {trainable_params} | Delta Params: {delta_params} | Total Params: {total_params}")
        
        # self.cls = nn.Linear(512, 10, bias=True)


    def forward(self, image):
        preds = self.delta_model.backbone_model(image)
        return preds

class LP1WithLora(nn.Module):
    def __init__(self, model):
        super().__init__()
        device = f"cuda:{str(config('device'))}" if torch.cuda.is_available() else "cpu"

        model_module = importlib.import_module("model")
        count_trainable_parameters_lora = model_module.count_trainable_parameters

        # self.model = model # this is the VIT-B model
        # self.clip_model = model.image_encoder.model.visual.to(device) # this is the backbone model
        self.clip_model = model.to(device) # this is the backbone model

        print("\nEntered LP1WITHLORA:")
        Visualization(self.clip_model).structure_graph()
        trainable, delta, total = count_trainable_parameters_lora(self.clip_model, None)
        print(f"[VIT-B/ENTERED MODEL] Trainable Params: {trainable} | Delta Params: {delta} | Total Params: {total}")


        print("\nActual Backbone Model:")
        Visualization(self.clip_model).structure_graph()
        trainable, delta, total = count_trainable_parameters_lora(self.clip_model, None)
        print(f"[BACKBONE MODEL] Trainable Params: {trainable} | Delta Params: {delta} | Total Params: {total}")

        # TODO need to change the modified modules and lora_r to improve the results 
        delta_model_temp = LoraModel(backbone_model=self.clip_model, 
                                modified_modules=['c_fc', 'c_proj'],
                                # modified_modules=['transformer.resblocks.0.mlp.c_proj', 
                                #                     'transformer.resblocks.1.mlp.c_proj', 
                                #                     'transformer.resblocks.2.mlp.c_proj', 
                                #                     'transformer.resblocks.3.mlp.c_proj', 
                                #                     'transformer.resblocks.4.mlp.c_proj', 
                                #                     'transformer.resblocks.5.mlp.c_proj', 
                                #                     'transformer.resblocks.6.mlp.c_proj', 
                                #                     'transformer.resblocks.7.mlp.c_proj', 
                                #                     'transformer.resblocks.8.mlp.c_proj', 
                                #                     'transformer.resblocks.9.mlp.c_proj',
                                #                     'transformer.resblocks.10.mlp.c_proj', 
                                #                     'transformer.resblocks.11.mlp.c_proj'],
                                lora_r=4)
        self.delta_model = delta_model_temp

        print("\nBackbone Model after adding LoRa:")
        self.delta_model.log()
        trainable_params, delta_params, total_params = count_trainable_parameters_lora(delta_model_temp.backbone_model, delta_model_temp.modified_modules)
        print(f"[BACKBONE MODEL & LoRa] Trainable Params: {trainable_params} | Delta Params: {delta_params} | Total Params: {total_params}")

        print("\nBackbone Model after adding LoRa, and FROZEN:")
        self.delta_model.freeze_module(exclude=["deltas", "ln_final"], set_state_dict=True)
        self.delta_model.log()
        trainable_params, delta_params, total_params  = count_trainable_parameters_lora(delta_model_temp.backbone_model, delta_model_temp.modified_modules)
        print(f"[BACKBONE MODEL & LoRa & FROZEN] Trainable Params: {trainable_params} | Delta Params: {delta_params} | Total Params: {total_params}")
        
        # self.cls = nn.Linear(512, 10, bias=True)


    def forward(self, image):
        preds = self.delta_model.backbone_model(image)
        return preds

class LP1WithLora_SVGD(nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        
        #device = f"cuda:{str(config('device'))}" if torch.cuda.is_available() else "cpu"
        
        device = "cuda:0"
        num_particles = opt

        # Define a regular expression that matches all layer names starting with 'lora'
        exclude_regex = re.compile(r'^lora.*')

        model_module = importlib.import_module("model")
        count_trainable_parameters_lora = model_module.count_trainable_parameters

        # self.clip_model = model.to(device)
        backbone = model.to(device)

        self.particles = []
        self.delta_models = []
        self.h_kernel = 0
  
        for i in range(num_particles):
            self.particles.append(copy.deepcopy(backbone))
        
        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)
            self.delta_models.append(LoraModel(backbone_model=self.particles[i], 
                                modified_modules=['c_fc', 'c_proj'],
                                lora_r=4))
            self.delta_models[i].log()
            self.delta_models[i].freeze_module(exclude=["deltas", "c_fc", "c_proj", "ln_final"], set_state_dict=True)
            print(f"\nBackbone Model {i} after adding LoRa + Freezing:")
            self.delta_models[i].log()

        #  check if lora is frozen
        for i in range(num_particles):
            for name, param in self.delta_models[i].backbone_model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)
                    print(param.requires_grad)


    def forward(self, image, **kwargs):
        logits, entropies, soft_out, stds = [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        for particle in self.particles:
            l = particle(image)
            sft = torch.softmax(l, 1)
            entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
            soft_out.append(sft)
            logits.append(l)
            # if return_entropy:
            #     l = torch.softmax(l, 1)
            # entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
        # disable average for now
        logits = torch.stack(logits).mean(0)
        # soft_out = torch.stack(soft_out).mean(0) 
        entropies = torch.stack(entropies).mean(0)
        # return logits, soft_out
        return logits


    def get_losses(self, image, labels, criterion, **kwargs):
        losses = []
        for particle in self.particles:
            l = particle(image)
            loss = criterion(l, labels)
            losses.append(loss)
        losses = torch.stack(losses).mean(0)
        # if torch.abs(losses - loss) > 1e-5:
        #     print("Losses are not equal")
        return losses

    def update_grads(self):
        if np.random.rand() < (1-float(config('p_update'))):
            return
        all_pgs = self.particles
        if self.h_kernel <= 0:
            self.h_kernel = 0.001  # 1
        dists = []
        alpha = float(config('alpha'))  # if t < 100 else 0.0
        new_parameters = [None] * len(all_pgs)

        for i in range(len(all_pgs)):
            new_parameters[i] = {}
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is None:
                    new_parameters[i][l] = None
                else:
                    new_parameters[i][l] = p.grad.data.new(
                        p.grad.data.size()).zero_()
            for j in range(len(all_pgs)):
                # if i == j:
                #     continue
                for l, params in enumerate(
                        zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                    p, p2 = params
                    if p.grad is None or p2.grad is None:
                        continue
                    if p is p2:
                        dists.append(0)
                        new_parameters[i][l] = new_parameters[i][l] + \
                            p.grad.data
                    else:
                        d = (p.data - p2.data).norm(2)
                        # if p is not p2:
                        dists.append(d.cpu().item())
                        kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
                        new_parameters[i][l] = (
                            ((new_parameters[i][l] + p2.grad.data) -
                             (d / self.h_kernel**2) * alpha) /
                            float(len(all_pgs))) * kij
        self.h_kernel = np.median(dists)
        self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)) + 1)
        for i in range(len(all_pgs)):
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is not None:
                    p.grad.data = new_parameters[i][l]

class LP1WithLora_SVGD_Curve(nn.Module):
    def __init__(self, model, opt, fix_points):
        super().__init__()
        device = f"cuda:{str(config('device'))}" if torch.cuda.is_available() else "cpu"
        num_particles = opt

        # Define a regular expression that matches all layer names starting with 'lora'
        exclude_regex = re.compile(r'^lora.*')

        model_module = importlib.import_module("model")
        count_trainable_parameters_lora = model_module.count_trainable_parameters

        # self.clip_model = model.to(device)
        backbone = model.to(device)

        self.particles = []
        self.delta_models = []
        self.h_kernel = 0
  
        for i in range(num_particles):
            self.particles.append(copy.deepcopy(backbone))
        
        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)
            self.delta_models.append(LoraModel(backbone_model=self.particles[i], 
                                modified_modules=['c_fc', 'c_proj'],
                                lora_r=4, fix_points=fix_points))
            self.delta_models[i].log()
            self.delta_models[i].freeze_module(exclude=["deltas", "c_fc", "c_proj", "ln_final"], set_state_dict=True)
            print(f"\nBackbone Model {i} after adding LoRa + Freezing:")
            self.delta_models[i].log()

        #  check if lora is frozen
        for i in range(num_particles):
            for name, param in self.delta_models[i].backbone_model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)
                    print(param.requires_grad)


    def forward(self, image, coeffs_t, **kwargs):
        logits, entropies, soft_out, stds = [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        for particle in self.delta_models:
            l = particle.backbone_model(image, coeffs_t)
            sft = torch.softmax(l, 1)
            entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
            soft_out.append(sft)
            logits.append(l)
            # if return_entropy:
            #     l = torch.softmax(l, 1)
            # entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
        # disable average for now
        logits = torch.stack(logits).mean(0)
        # soft_out = torch.stack(soft_out).mean(0) 
        entropies = torch.stack(entropies).mean(0)
        # return logits, soft_out
        return logits


    def get_losses(self, image, labels, criterion, **kwargs):
        losses = []
        for particle in self.particles:
            l = particle(image)
            loss = criterion(l, labels)
            losses.append(loss)
        losses = torch.stack(losses).mean(0)
        # if torch.abs(losses - loss) > 1e-5:
        #     print("Losses are not equal")
        return losses

    def update_grads(self):
        if np.random.rand() < (1-float(config('p_update'))):
            return
        all_pgs = self.particles
        if self.h_kernel <= 0:
            self.h_kernel = 0.001  # 1
        dists = []
        alpha = float(config('alpha'))  # if t < 100 else 0.0
        new_parameters = [None] * len(all_pgs)

        for i in range(len(all_pgs)):
            new_parameters[i] = {}
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is None:
                    new_parameters[i][l] = None
                else:
                    new_parameters[i][l] = p.grad.data.new(
                        p.grad.data.size()).zero_()
            for j in range(len(all_pgs)):
                # if i == j:
                #     continue
                for l, params in enumerate(
                        zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                    p, p2 = params
                    if p.grad is None or p2.grad is None:
                        continue
                    if p is p2:
                        dists.append(0)
                        new_parameters[i][l] = new_parameters[i][l] + \
                            p.grad.data
                    else:
                        d = (p.data - p2.data).norm(2)
                        # if p is not p2:
                        dists.append(d.cpu().item())
                        kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
                        new_parameters[i][l] = (
                            ((new_parameters[i][l] + p2.grad.data) -
                             (d / self.h_kernel**2) * alpha) /
                            float(len(all_pgs))) * kij
        self.h_kernel = np.median(dists)
        self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)) + 1)
        for i in range(len(all_pgs)):
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is not None:
                    p.grad.data = new_parameters[i][l]

class BayesWrap(nn.Module):
    def __init__(self, NET, opt):
        super().__init__()

        num_particles = int(config('opt'))
        self.h_kernel = 0
        self.particles = []

        for i in range(num_particles):
            self.particles.append(copy.deepcopy(NET))


        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)

        # logging.info("num particles: %d" % len(self.particles))
        print(f"num particles: {len(self.particles)}")

    def sample_particle(self):
        return self.particles[np.random.randint(0, len(self.particles))]

    def get_particle(self, index):
        return self.particles[index]

    def forward_q(self, q_rep, return_entropy=True):
        logits, entropies = [], []
        for particle in self.particles:
            l = particle.classifier(q_rep)
            if return_entropy:
                l = torch.softmax(l, 0)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
            logits.append(l)
        logits = torch.stack(logits).mean(0)
        if return_entropy:
            entropies = torch.stack(entropies).mean(0)
            return logits, entropies
        return logits

    def forward(self, x, **kwargs):
        logits, entropies, soft_out, stds = [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        for particle in self.particles:
            l = particle(x)
            sft = torch.softmax(l, 1)
            soft_out.append(sft)
            logits.append(l)
            if return_entropy:
                l = torch.softmax(l, 1)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
        logits = torch.stack(logits).mean(0)
        stds = torch.stack(soft_out).std(0)
        soft_out = torch.stack(soft_out).mean(0)
        if return_entropy:
            entropies = torch.stack(entropies).mean(0)
            return logits, entropies, soft_out, stds
        return logits, soft_out

    def update_grads(self):
        if np.random.rand() < 1 - float(config('p_update')):
            return
        all_pgs = self.particles
        if self.h_kernel <= 0:
            self.h_kernel = 0.1  # 1
        dists = []
        alpha = float(config('alpha'))
        new_parameters = [None] * len(all_pgs)

        for i in range(len(all_pgs)):
            new_parameters[i] = {}
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is None:
                    new_parameters[i][l] = None
                else:
                    new_parameters[i][l] = p.grad.data.new(
                        p.grad.data.size()).zero_()
            for j in range(len(all_pgs)):
                # if i == j:
                #     continue
                for l, params in enumerate(
                        zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                    p, p2 = params
                    if p.grad is None or p2.grad is None:
                        continue
                    if p is p2:
                        dists.append(0)
                        new_parameters[i][l] = new_parameters[i][l] + \
                            p.grad.data
                    else:
                        d = (p.data - p2.data).norm(2)
                        # if p is not p2:
                        dists.append(d.cpu().item())
                        kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
                        new_parameters[i][l] = (
                            ((new_parameters[i][l] + p2.grad.data) -
                             (d / self.h_kernel**2) * alpha) /
                            float(len(all_pgs))) * kij
        self.h_kernel = np.median(dists)
        self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)) + 1)
        for i in range(len(all_pgs)):
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is not None:
                    p.grad.data = new_parameters[i][l]

        def update_gradiants(all_pgs):

            if np.random.rand() < 0.2:
                return
            self.h_kernel = 0.1

            dists = []
            alpha = 0.01  # if t < 100 else 0.0
            new_parameters = [None] * len(all_pgs)

            for i in range(len(all_pgs)):
                new_parameters[i] = {}
                for l, p in enumerate(all_pgs[i].parameters()):
                    if p.grad is None:
                        new_parameters[i][l] = None
                    else:
                        new_parameters[i][l] = p.grad.data.new(
                            p.grad.data.size()).zero_()
                for j in range(len(all_pgs)):
                    # if i == j:
                    #     continue
                    for l, params in enumerate(
                            zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                        p, p2 = params
                        if p.grad is None or p2.grad is None:
                            continue
                        if p is p2:
                            dists.append(0)
                            new_parameters[i][l] = new_parameters[i][l] + \
                                p.grad.data
                        else:
                            d = (p.data - p2.data).norm(2)
                            # if p is not p2:
                            dists.append(d.cpu().item())
                            kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
                            new_parameters[i][l] = (
                                ((new_parameters[i][l] + p2.grad.data) -
                                    (d / self.h_kernel**2) * alpha) /
                                float(len(all_pgs))) * kij
            self.h_kernel = np.median(dists)
            self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)) + 1)
            for i in range(len(all_pgs)):
                for l, p in enumerate(all_pgs[i].parameters()):
                    if p.grad is not None:
                        p.grad.data = new_parameters[i][l]


def generate_adapter_model(particles):

# Create some dummy inputs
    dummy_inputs = torch.randn(1, 3, 32, 32)
    dummy_inputs = dummy_inputs.to(torch.long)
    # TODO: try to pass this dummy_inputs to particle to see if errored

    model_module = importlib.import_module("model")

    delta_models = []
    for i, particle in enumerate(particles):
     
    #  Set the dummy inputs for the backbone model
        setattr(particle, "dummy_inputs", dummy_inputs)
        delta_models.append(LowRankAdapterModel(backbone_model=particle, 
                            modified_modules=['c_fc', 'c_proj'],
                            ))
        delta_models[i].log()
        delta_models[i].freeze_module(exclude=["deltas", "ln_final"], set_state_dict=True)
        print(f"\nBackbone Model {i} after adding LoRa + Freezing:")
        delta_models[i].log()

    print(f'number of individual delta models are: {len(delta_models)}')  
    
    return delta_models

def generate_lora_particles(particles, rank):

    exclude_regex = re.compile(r'^lora.*')
    model_module = importlib.import_module("model")
    count_trainable_parameters_lora = model_module.count_trainable_parameters

    print("\nEntered LP1WITHLORA:")
    Visualization(particles[0]).structure_graph()
    trainable, delta, total = count_trainable_parameters_lora(particles[0], None)
    print(f"[VIT-B/ENTERED MODEL] Trainable Params: {trainable} | Delta Params: {delta} | Total Params: {total}")

    print("\nActual Backbone Model:")
    Visualization(particles[0]).structure_graph()
    trainable, delta, total = count_trainable_parameters_lora(particles[0], None)
    print(f"[BACKBONE MODEL] Trainable Params: {trainable} | Delta Params: {delta} | Total Params: {total}")

    delta_models = []
    for i, particle in enumerate(particles):
     
        delta_models.append(LoraModel(backbone_model=particle, 
                            modified_modules=['c_fc', 'c_proj'],
                            lora_r=rank))
        delta_models[i].log()
        delta_models[i].freeze_module(exclude=["deltas"], set_state_dict=True)
        print(f"\nBackbone Model {i} after adding LoRa + Freezing:")
        delta_models[i].log()

    print(f'number of individual delta models are: {len(delta_models)}')  
    
    return delta_models

def train_model_wrap(particles, trainloaders, valloader, noise_std, config):

    criterion = nn.CrossEntropyLoss()

    best_losses = [float('inf')] * len(particles)
    best_val_accuracy = [float('inf')] * len(particles)

    learning_rates = [0.001, 0.002, 0.0005, 0.0001, 0.0008]

    optimizers = [optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr) for model, lr in zip(particles, learning_rates)]


    for epoch in range(int(config('num_epochs'))):
        
        accumulated_losses = [0.0] * len(particles)
        num_batches = len(next(iter(trainloaders)))

        for j,batches in enumerate(zip(*trainloaders)):
            inputs_list = [batch[0] for batch in batches]
            targets_list = [batch[1] for batch in batches]
            for i, (model, imgs, lbls) in enumerate(zip(particles, inputs_list, targets_list)):
                imgs, labels = imgs.cuda(), lbls.cuda()

                optimizers[i].zero_grad()

                logits = model(imgs)

                loss = criterion(logits, labels)
                loss.backward()
                accumulated_losses[i] += loss.item()
            print(f'\rProcessing batch {j+1}/{num_batches}', end='')

            update_gradiants(particles)

            for optimizer in optimizers:
                optimizer.step()

        average_losses = [loss_sum / num_batches for loss_sum in accumulated_losses]
        for i, avg_loss in enumerate(average_losses):
            print(f"Epoch {epoch}, Model {i}, Average Epoch Loss: {avg_loss}")

    
        with torch.no_grad():
            for i,model in enumerate(particles):

                correct = 0
                total = 0
                losses_eval, step2 = 0., 0.
                for img, lbls,_ in valloader:
                    img, label = img.cuda(), lbls.cuda()

                    logits = model(img)
                    loss_val = criterion(logits, label)
                    losses_eval += loss_val.item()
                    _, predicted = torch.max(logits, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    step2 += 1

                accuracy = correct / total
                loss_val_final = losses_eval / step2
                print(f'[Epoch: {epoch}], val_acc_{i}: {accuracy:.4f}, val_loss_{i}: {loss_val_final:.4f}')
                
                # 3. Save Models with Best Validation Loss
                model_idx = particles.index(model)
                if loss_val_final < best_losses[model_idx]:
                    best_losses[model_idx] = loss_val_final
                    best_val_accuracy[model_idx] = accuracy
                    best_epoch = epoch
                    best_model = copy.deepcopy(model.state_dict())

                    best_model_path = f"mdl-wrap/best_model_{i}_noise_std_{noise_std}.pt"
                    torch.save(best_model, best_model_path)
                    print(f'Best model {i} at epoch {best_epoch} has been saved')

    with open("mdl-wrap/best_val_accuracy.txt", "w") as file:
    # Write each accuracy value to the file, one value per line
        for i,accuracy in enumerate(best_val_accuracy):
            file.write(f"best val_acc for model {i} is {accuracy}\n")
    print('finished')        

def generate_freezed_particles(mdl , num_ensemble):

    classification_head = get_classification_head()
    image_encoder = ImageEncoder(mdl)
    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()

    NET = NET.cuda()
    particles = []
    for i in range(num_ensemble):
        particles.append(copy.deepcopy(NET))

    print(f'number of individual models: {len(particles)}')  
    
    return particles  

def train_model_wrap_cifar(particles, trainloaders, valloader, noise_std, config):

    criterion = nn.CrossEntropyLoss()

    best_losses = [float('inf')] * len(particles)
    best_val_accuracy = [float('inf')] * len(particles)

    learning_rates = [0.001, 0.002, 0.0005, 0.0001, 0.0008]

    optimizers = [optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr) for model, lr in zip(particles, learning_rates)]


    for epoch in range(int(config('num_epochs'))):
        
        accumulated_losses = [0.0] * len(particles)
        num_batches = len(next(iter(trainloaders)))

        for j,batches in enumerate(zip(*trainloaders)):
            inputs_list = [batch[0] for batch in batches]
            targets_list = [batch[1] for batch in batches]
            for i, (model, imgs, lbls) in enumerate(zip(particles, inputs_list, targets_list)):
                imgs, labels = imgs.cuda(), lbls.cuda()

                optimizers[i].zero_grad()

                logits = model(imgs)

                loss = criterion(logits, labels)
                loss.backward()
                accumulated_losses[i] += loss.item()
            print(f'\rProcessing batch {j+1}/{num_batches}', end='')

            update_gradiants(particles)

            for optimizer in optimizers:
                optimizer.step()

        average_losses = [loss_sum / num_batches for loss_sum in accumulated_losses]
        for i, avg_loss in enumerate(average_losses):
            print(f"Epoch {epoch}, Model {i}, Average Epoch Loss: {avg_loss}")

    
        with torch.no_grad():
            for i,model in enumerate(particles):

                correct = 0
                total = 0
                losses_eval, step2 = 0., 0.
                for img, lbls in valloader:
                    img, label = img.cuda(), lbls.cuda()

                    logits = model(img)
                    loss_val = criterion(logits, label)
                    losses_eval += loss_val.item()
                    _, predicted = torch.max(logits, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    step2 += 1

                accuracy = correct / total
                loss_val_final = losses_eval / step2
                print(f'[Epoch: {epoch}], val_acc_{i}: {accuracy:.4f}, val_loss_{i}: {loss_val_final:.4f}')
                
                # 3. Save Models with Best Validation Loss
                model_idx = particles.index(model)
                if loss_val_final < best_losses[model_idx]:
                    best_losses[model_idx] = loss_val_final
                    best_val_accuracy[model_idx] = accuracy
                    best_epoch = epoch
                    best_model = copy.deepcopy(model.state_dict())

                    best_model_path = f"Model/best_model_{i}_noise_std_{noise_std}.pt"
                    torch.save(best_model, best_model_path)
                    print(f'Best model {i} at epoch {best_epoch} has been saved')

    with open("Model/best_val_accuracy.txt", "w") as file:
    # Write each accuracy value to the file, one value per line
        for i,accuracy in enumerate(best_val_accuracy):
            file.write(f"best val_acc for model {i} is {accuracy}\n")
    print('finished')        

def update_gradiants(all_pgs, h_kernel=None):

    if np.random.rand() < 0.95:
        return

    if h_kernel is None or h_kernel <= 0:
        h_kernel = 0.001  # 1
    dists = []
    alpha = 0.01  # if t < 100 else 0.0
    new_parameters = [None] * len(all_pgs)

    for i in range(len(all_pgs)):
        new_parameters[i] = {}
        for l, p in enumerate(all_pgs[i].parameters()):
            if p.grad is None:
                new_parameters[i][l] = None
            else:
                new_parameters[i][l] = p.grad.data.new(
                    p.grad.data.size()).zero_()
        for j in range(len(all_pgs)):
            # if i == j:
            #     continue
            for l, params in enumerate(
                    zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                p, p2 = params
                if p.grad is None or p2.grad is None:
                    continue
                if p is p2:
                    dists.append(0)
                    new_parameters[i][l] = new_parameters[i][l] + \
                        p.grad.data
                else:
                    d = (p.data - p2.data).norm(2)
                    # if p is not p2:
                    dists.append(d.cpu().item())
                    kij = torch.exp(-(d**2) / h_kernel**2 / 2)
                    new_parameters[i][l] = (
                        ((new_parameters[i][l] + p2.grad.data) -
                            (d / h_kernel**2) * alpha) /
                        float(len(all_pgs))) * kij
    h_kernel = np.median(dists)
    h_kernel = np.sqrt(0.5 * h_kernel / np.log(len(all_pgs)) + 1)
    for i in range(len(all_pgs)):
        for l, p in enumerate(all_pgs[i].parameters()):
            if p.grad is not None:
                p.grad.data = new_parameters[i][l]

    return h_kernel         


def generate_lora_model(mdl, num_ensemble):
    classification_head = get_classification_head()
    image_encoder = ImageEncoder(mdl)
    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()
    
    model = LP1WithLora_SVGD(NET, num_ensemble)
    Visualization(model).structure_graph()
    return model

