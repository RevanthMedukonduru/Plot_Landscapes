import sys
import torch
import math
import copy
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from decouple import config
from utils import add_noise_to_parameters
import numpy as np
import os
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
from bayes_wrap import BayesWrap, LP1WithLora, LP1WithAdapter, LP1WithLora_SVGD, SVGD
import torch.optim.lr_scheduler as lr_scheduler
from utils import cosine_lr

from opendelta.utils.inspect import inspect_module_statistics
from bigmodelvis import Visualization

"""
from opendelta import Visualization
from opendelta import LowRankAdapterModel, AdapterModel
from opendelta import LoraModel # use lora as an example, others are same
"""

# check if config file is loaded else initialize it
try:
    config('dataset_name')
except:
    config('device', default='0')
device = f"cuda:{config('device')}" if torch.cuda.is_available() else "cpu"

def train_model_camelyon(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):
    """this function trains the model and returns the losses and the entropies of the test set"""

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()


    opt = config('opt')
    model = BayesWrap(NET, opt)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=float(config('learning_rate')))


    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, step = 0., 0.
        for i, (img, labels, metadata) in enumerate(train_loader):
            
            img, labels = img.cuda(), labels.cuda()

            optimizer.zero_grad()

            kwargs = {"return_entropy": False}
            logits, soft_out = model(img, **kwargs)

            loss = criterion(logits, labels)
            loss.backward()
            model.update_grads()

            losses += loss.item()
            step += 1
            
            optimizer.step()     
            optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        

        loss_epoch = losses / step
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, step2 = 0., 0.
        for img, text, metadata in validation_loader:
            img, text = img.cuda(), text.cuda()
            logits, soft_out = model(img, **kwargs)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(soft_out, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            step2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / step2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        # Save checkpoint if the current validation loss is the best
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"


    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs=[], []
    i = 0
    with torch.no_grad():

        for images, labels, metadata in test_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out = model(img, **kwargs)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy()) 
            all_softs.extend(soft_out.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    all_scores_train = []
    all_labels_train = []
    all_entropies_train=[]

    i = 0
    with torch.no_grad():

        for images, labels, metadata in train_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out = model(img, **kwargs)

            predicted = torch.argmax(logits, dim=1)
            all_scores_train.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels_train.extend(labels.numpy())  # Extend the list with true labels
            all_entropies_train.extend(entropies.cpu().numpy())
            print(f'\r calculating entropies for train {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_scores_train = np.array(all_scores_train).tolist()
        all_labels_train = np.array(all_labels_train).tolist()
        all_entropies_train = np.array(all_entropies_train).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test": all_softs,
                     "all_labels_train": all_labels_train,
                     "all_scores_train": all_scores_train,
                     "all_entropies_train": all_entropies_train}

    run_name = config('run_name')
    labels_info_path = f"Results/{run_name}/entropies.json"
    if not os.path.exists(f"Results/{run_name}"):
        os.makedirs(f"Results/{run_name}")
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    # Save best model
    # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"
    # torch.save(best_model, best_model_path)

    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels


def train_model_cifar(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    NET = ImageClassifier(image_encoder, classification_head)
    NET.freeze_head()

    opt = config('opt')
    model = BayesWrap(NET, opt)

    trainable = count_trainable_parameters(model)
    print("trainable parameters of normal model: ", trainable)
    model = model.cuda()

    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=float(config('learning_rate')))

    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, step = 0., 0.
        for i, (img, labels) in enumerate(train_loader):
            
            img, labels = img.cuda(), labels.cuda()

            optimizer.zero_grad()

            kwargs = {"return_entropy": False}
            logits, soft_out = model(img, **kwargs)

            loss = criterion(logits, labels)
            loss.backward()
            model.update_grads()

            losses += loss.item()
            step += 1
            
            optimizer.step()     
            # optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / step
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, step2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.cuda(), text.cuda()
            logits, soft_out = model(img, **kwargs)
        
            loss_val = criterion(soft_out, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            step2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / step2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"


    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out, stds = model(img, **kwargs)

            predicted = torch.argmax(soft_out, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(soft_out.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    all_scores_train = []
    all_labels_train = []
    all_entropies_train=[]

    i = 0
    with torch.no_grad():

        for images, labels in train_loader:
            img , text = images.cuda(), labels.cuda()
            kwargs = {"return_entropy": True}
            logits, entropies, soft_out, _ = model(img, **kwargs)

            predicted = torch.argmax(soft_out, dim=1)
            all_scores_train.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels_train.extend(labels.numpy())  # Extend the list with true labels
            all_entropies_train.extend(entropies.cpu().numpy())
            print(f'\r calculating entropies for train {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()
        all_scores_train = np.array(all_scores_train).tolist()
        all_labels_train = np.array(all_labels_train).tolist()
        all_entropies_train = np.array(all_entropies_train).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds,
                     "all_labels_train": all_labels_train,
                     "all_scores_train": all_scores_train,
                     "all_entropies_train": all_entropies_train}

    run_name = config('run_name')
    labels_info_path = f"Results/{run_name}/entropies.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels


def count_trainable_parameters(model, modified_modules=None): 
    """
    Counts the no of delta parameters, trainable parameters and total parameters in a LoRa model.

    Args:
        model: A LoRa model.
        modified_modules: A list of modified modules in the model.
    
    Returns:
        delta_parameters: The number of delta parameters in the model.
        trainable_parameters: The number of trainable parameters in the model.
        total_parameters: The total number of parameters in the model.
    """
    stat = inspect_module_statistics(model, modified_modules)
    delta_parameters = stat["delta_parameters"]
    trainable_parameters = stat["trainable_parameters"]
    total_params = stat["total_parameters"]
    return trainable_parameters, delta_parameters, total_params


"""
from torchsummary import summary
"""
from opendelta import AdapterModel

def train_model_cifar_Adapter(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):
    """train a model using Adapter"""

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()

    # opt = config('opt')
    # model = BayesWrap(NET, opt)
    # model = LP1WithLora(net)


    # print how much trainable paramters of LoRa model
    # summary(model, input_size=(3, 32, 32))
    # trainable = count_trainable_parameters(model)
    # print("trainable parameters of LoRa model: ", trainable)

    model = net
    model = model.to(device)

    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params, lr=float(config('learning_rate')))
    # change to Adam optimizer
    # optimizer = optim.Adam(params, lr=float(config('learning_rate')))
    optimizer = torch.optim.AdamW(params, lr=float(config('learning_rate')), weight_decay=float(config('Weight_decay')))
    
    scheduler = cosine_lr(
        optimizer,
        float(config('learning_rate')),
        int(config("warmup_length")),
        int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
    )
    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, tep = 0., 0.
        for i, (img, labels) in enumerate(train_loader):

            step = (
                i // int(config('num_grad_accumulation'))
                + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
            )
            
            img, labels = img.to(device), labels.to(device)
            model = LP1WithAdapter(model, img)

            optimizer.zero_grad()

            # kwargs = {"return_entropy": False}
            logits = model(img)

            loss = criterion(logits, labels)
            loss.backward()
            # model.update_grads()

            losses += loss.item()
            tep += 1
            
            if (i + 1) % int(config('num_grad_accumulation')) == 0:
                scheduler(step)

                optimizer.step()     
            # optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / tep
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, tep2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.to(device), text.to(device)
            logits = model(img)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            tep2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / tep2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"
        
        # save the best model
        if epoch == best_epoch:
            print("saving the best model")
            if not os.path.exists(f"Model/{config('run_name')}"):
                os.makedirs(f"Model/{config('run_name')}")
            torch.save(model.state_dict(), f"Model/{config('run_name')}/best_model_noise_std_{noise_std}.pt")
        


    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            kwargs = {"return_entropy": True}
            # logits, entropies, soft_out, stds = model(img, **kwargs)
            logits = model(img)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            # all_entropies.extend(entropies.cpu().numpy())
            # all_softs.extend(soft_out.cpu().numpy())
            # all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    all_scores = np.array(all_scores).tolist()
    all_labels = np.array(all_labels).tolist()


    # all_scores_train = []
    # all_labels_train = []
    # all_entropies_train=[]

    # i = 0
    # with torch.no_grad():

    #     for images, labels in train_loader:
    #         img , text = images.to(device), labels.to(device)
    #         kwargs = {"return_entropy": True}
    #         # logits, entropies, soft_out, _ = model(img, **kwargs)
    #         logits = model(img)

    #         # predicted = torch.argmax(soft_out, dim=1)
    #         predicted = torch.argmax(logits, dim=1)
    #         all_scores_train.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
    #         all_labels_train.extend(labels.numpy())  # Extend the list with true labels
    #         # all_entropies_train.extend(entropies.cpu().numpy())
    #         print(f'\r calculating entropies for train {i}', end='')
    #         i +=1

    #     # Convert the lists of scores and labels to NumPy arrays
    #     all_scores = np.array(all_scores).tolist()
    #     all_labels = np.array(all_labels).tolist()
    #     # all_entropies = np.array(all_entropies).tolist()
    #     # all_softs = np.array(all_softs).tolist()
    #     # all_stds = np.array(all_stds).tolist()
    #     all_scores_train = np.array(all_scores_train).tolist()
    #     all_labels_train = np.array(all_labels_train).tolist()
    #     # all_entropies_train = np.array(all_entropies_train).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores 
                    #  "all_entropies_test":  all_entropies,
                    #  "all_softs_test":  all_softs,
                    #  "all_std_test": all_stds,
                    #  "all_labels_train": all_labels_train,
                    #  "all_scores_train": all_scores_train,
                    #  "all_entropies_train": all_entropies_train
                     }
    run_name = config('run_name')
    labels_info_path = f"Results/{run_name}/entropies.json"
    if not os.path.exists(f"Results/{run_name}"):
        os.makedirs(f"Results/{run_name}")
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels



def train_model_cifar_LoRa(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):
    """train a model using LoRa"""

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()

    opt = config('opt')
    # model = BayesWrap(NET, opt)
    model = LP1WithLora(net)

    # print how much trainable paramters of LoRa model
    # summary(model, input_size=(3, 32, 32))
    trainable = count_trainable_parameters(model)
    print("trainable parameters of LoRa model: ", trainable)

    model = model.to(device)

    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params, lr=float(config('learning_rate')))
    # change to Adam optimizer
    # optimizer = optim.Adam(params, lr=float(config('learning_rate')))
    optimizer = torch.optim.AdamW(params, lr=float(config('learning_rate')), weight_decay=float(config('Weight_decay')))
    
    scheduler = cosine_lr(
        optimizer,
        float(config('learning_rate')),
        int(config("warmup_length")),
        int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
    )
    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, tep = 0., 0.
        for i, (img, labels) in enumerate(train_loader):

            step = (
                i // int(config('num_grad_accumulation'))
                + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
            )
            
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()

            # kwargs = {"return_entropy": False}
            logits = model(img)

            loss = criterion(logits, labels)
            loss.backward()
            # model.update_grads()

            losses += loss.item()
            tep += 1
            
            if (i + 1) % int(config('num_grad_accumulation')) == 0:
                scheduler(step)

                optimizer.step()     
            # optimizer.zero_grad()
            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / tep
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, tep2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.to(device), text.to(device)
            logits = model(img)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            tep2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / tep2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"
        
        # save the best model
        if epoch == best_epoch:
            print("saving the best model")
            if not os.path.exists(f"Model/{config('run_name')}"):
                os.makedirs(f"Model/{config('run_name')}")
            torch.save(model.state_dict(), f"Model/{config('run_name')}/best_model_noise_std_{noise_std}.pt")
        


    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            kwargs = {"return_entropy": True}
            # logits, entropies, soft_out, stds = model(img, **kwargs)
            logits = model(img)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            # all_entropies.extend(entropies.cpu().numpy())
            # all_softs.extend(soft_out.cpu().numpy())
            # all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1

    all_scores = np.array(all_scores).tolist()
    all_labels = np.array(all_labels).tolist()


    # all_scores_train = []
    # all_labels_train = []
    # all_entropies_train=[]

    # i = 0
    # with torch.no_grad():

    #     for images, labels in train_loader:
    #         img , text = images.to(device), labels.to(device)
    #         kwargs = {"return_entropy": True}
    #         # logits, entropies, soft_out, _ = model(img, **kwargs)
    #         logits = model(img)

    #         # predicted = torch.argmax(soft_out, dim=1)
    #         predicted = torch.argmax(logits, dim=1)
    #         all_scores_train.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
    #         all_labels_train.extend(labels.numpy())  # Extend the list with true labels
    #         # all_entropies_train.extend(entropies.cpu().numpy())
    #         print(f'\r calculating entropies for train {i}', end='')
    #         i +=1

    #     # Convert the lists of scores and labels to NumPy arrays
    #     all_scores = np.array(all_scores).tolist()
    #     all_labels = np.array(all_labels).tolist()
    #     # all_entropies = np.array(all_entropies).tolist()
    #     # all_softs = np.array(all_softs).tolist()
    #     # all_stds = np.array(all_stds).tolist()
    #     all_scores_train = np.array(all_scores_train).tolist()
    #     all_labels_train = np.array(all_labels_train).tolist()
    #     # all_entropies_train = np.array(all_entropies_train).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores 
                    #  "all_entropies_test":  all_entropies,
                    #  "all_softs_test":  all_softs,
                    #  "all_std_test": all_stds,
                    #  "all_labels_train": all_labels_train,
                    #  "all_scores_train": all_scores_train,
                    #  "all_entropies_train": all_entropies_train
                     }
    run_name = config('run_name')
    labels_info_path = f"Results/{run_name}/entropies.json"
    if not os.path.exists(f"Results/{run_name}"):
        os.makedirs(f"Results/{run_name}")
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels


def train_model_cifar_BNN(mdl, train_loader, validation_loader, test_loader, noise_std, j, config):
    """train a model using LoRa applying SVGD for BNN approximation"""

    classification_head = get_classification_head()

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)

    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()

    opt = config('opt')
    
    model = SVGD(net, opt)
    print("\nArchitecture recieved from SVGD:")
    Visualization(model).structure_graph()
    trainable, delta, total = count_trainable_parameters(model, None)
    print(f"[LORA_SVGD FINAL] Trainable parameters: {trainable}, delta parameters: {delta}, total parameters: {total}")

    model = model.to(device)


    ## freeze all parameters except deltas
    # for name, param in model.named_parameters():
    #     if 'lora' not in name:
    #         param.requires_grad = False

    # Visualize the model
    # print("\nFinal model architecture, FROZEN:")
    # Visualization(model).structure_graph()
    # trainable, delta, total = count_trainable_parameters(model, None)
    # print(F"[LORA_SVGD FROZEN FINAL] Trainable parameters: {trainable}, delta parameters: {delta}, total parameters: {total}")

    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=float(config('learning_rate')))
    optimizer = torch.optim.AdamW(params, lr=float(config('learning_rate')), weight_decay=float(config('decay_rate')))

    scheduler = cosine_lr(
        optimizer,
        float(config('learning_rate')),
        int(config("warmup_length")),
        int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
    )    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)



    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, tep = 0., 0.
        for i, (img, labels) in enumerate(train_loader):
            step = (
                i // int(config('num_grad_accumulation'))
                + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
            )            
            
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()

            # kwargs = {"return_entropy": True}
            # logits, soft_out = model(img)
            loss = model.get_losses(img, labels, criterion)
            # print("Logits", len(logits), logits[0].shape, logits[1].shape)
            
            loss.backward()
            model.update_grads()

            losses += loss.item()
            tep += 1

            if (i + 1) % int(config('num_grad_accumulation')) == 0:
                scheduler(step)
                optimizer.step()  

            # optimizer.step()            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / tep
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, tep2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.to(device), text.to(device)
            logits,_  = model(img)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            tep2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / tep2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"


    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            kwargs = {"return_entropy": True}
            # logits, entropies, soft_out, stds = model(img, **kwargs)
            logits, entropies = model(img)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            # all_softs.extend(soft_out.cpu().numpy())
            # all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1


        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies
                    #  "all_softs_test":  all_softs,
                    #  "all_std_test": all_stds,
                    #  "all_labels_train": all_labels_train,
                    #  "all_scores_train": all_scores_train,
                    #  "all_entropies_train": all_entropies_train
                     }
    run_name = config('run_name')
    labels_info_path = "Results/{run_name}"
    if not os.path.exists(labels_info_path):
        os.makedirs(labels_info_path)
    labels_info_path = os.path.join(labels_info_path, 'entropies.json')
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels


def train_model_cifar_LoRa_BNN(mdl, curve_mdl, train_loader, validation_loader, test_loader, noise_std, j, config):
    """train a model using LoRa applying SVGD for BNN approximation"""

    classification_head = get_classification_head()
    #print("Classification head: ", classification_head)
    

    if config("linear").lower() == 'true':
        image_encoder = LinearizedImageEncoder(mdl, keep_lang=False)
        print('model is loaded in linearized mode for fine-tuning')

    else:
        image_encoder = ImageEncoder(mdl)#, keep_lang=False)
        curve_image_encoder = ImageEncoder(curve_mdl)#, keep_lang=False)

    if noise_std != 0:
        add_noise_to_parameters(image_encoder, noise_std)
        add_noise_to_parameters(curve_image_encoder, noise_std)

    net = ImageClassifier(image_encoder, classification_head)
    net.freeze_head()
    
    curve_net = ImageClassifier(curve_image_encoder, classification_head)
    curve_net.freeze_head()

    opt = config('opt')
    
    model = LP1WithLora_SVGD(net, opt)
    print("\nArchitecture recieved from LP1WithLoRa_SVGD:")
    Visualization(model).structure_graph()
    trainable, delta, total = count_trainable_parameters(model, None)
    print(f"[LORA_SVGD FINAL] Trainable parameters: {trainable}, delta parameters: {delta}, total parameters: {total}")

    isCurveModel = True
    curve_model = LP1WithLora_SVGD(curve_net, opt, isCurveModel)
    print("\nArchitecture recieved from LP1WithLoRa_SVGD Curve Model:")
    Visualization(curve_model).structure_graph()
    trainable, delta, total = count_trainable_parameters(curve_model, None)
    print(f"[LORA_SVGD CURVE FINAL] Trainable parameters: {trainable}, delta parameters: {delta}, total parameters: {total}")
    
    model = model.to(device)
    curve_model = curve_model.to(device)

    ## freeze all parameters except deltas
    # for name, param in model.named_parameters():
    #     if 'lora' not in name:
    #         param.requires_grad = False

    # Visualize the model
    print("\nFinal model architecture, FROZEN:")
    Visualization(model).structure_graph()
    trainable, delta, total = count_trainable_parameters(model, None)
    print(F"[LORA_SVGD FROZEN FINAL] Trainable parameters: {trainable}, delta parameters: {delta}, total parameters: {total}")
    
    # temperature_factor = float(config('temperature_factor'))
    criterion = nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=float(config('learning_rate')))
    optimizer = torch.optim.AdamW(params, lr=float(config('learning_rate')), weight_decay=float(config('decay_rate')))

    scheduler = cosine_lr(
        optimizer,
        float(config('learning_rate')),
        int(config("warmup_length")),
        int(config('num_epochs')) * int(config('batch_size')) // int(config('num_grad_accumulation'))
    )    

    loss_values_epoch = []
    loss_values_val = []
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    ti = len(train_loader)

    # FIXME why the code is still consuming high GPU memory?


    for epoch in range(int(config('num_epochs'))):
        # Training
        model.train()


        losses, tep = 0., 0.
        for i, (img, labels) in enumerate(train_loader):
            step = (
                i // int(config('num_grad_accumulation'))
                + epoch * int(config('batch_size')) // int(config('num_grad_accumulation'))
            )            
            
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()

            # kwargs = {"return_entropy": True}
            # logits, soft_out = model(img)
            loss = model.get_losses(img, labels, criterion)
            # print("Logits", len(logits), logits[0].shape, logits[1].shape)
            
            loss.backward()
            model.update_grads()

            losses += loss.item()
            tep += 1

            if (i + 1) % int(config('num_grad_accumulation')) == 0:
                scheduler(step)
                optimizer.step()  

            # optimizer.step()            
            print(f"\r[Epoch: {epoch}], iter:{i+1} from {ti}, loss: {loss.item():.4f}", end='')
        
        

        loss_epoch = losses / tep
        print(f' loss epoch: {loss_epoch:.4f}')

        # Evaluation
        model.eval()

        correct = 0
        total = 0
        losses_eval, tep2 = 0., 0.
        for img, text in validation_loader:
            img, text = img.to(device), text.to(device)
            logits, _  = model(img)
        
            loss_val = criterion(logits, text)
            losses_eval += loss_val.item()
            _, predicted = torch.max(logits, 1)
            total += text.size(0)
            correct += (predicted == text).sum().item()
            tep2 += 1

        accuracy = correct / total
        loss_val_final = losses_eval / tep2
        print(f'[Epoch: {epoch}], val_accuracy: {accuracy:.4f}, val_loss: {loss_val_final:.4f}')

        loss_values_val.append(loss_val_final)
        loss_values_epoch.append(loss_epoch)

        
        if loss_val_final < best_val_loss:
            best_val_loss = loss_val_final
            best_val_accuracy = accuracy
            best_epoch = epoch
            train_loss_best_epoch = loss_epoch
            # best_model = copy.deepcopy(model.state_dict())
            # best_model_path = f"Model/best_model_{j}_noise_std_{noise_std}.pt"


    model.eval()  

    # Evaluation loop
    all_scores = []
    all_labels = []
    all_entropies, all_softs, all_stds=[], [], []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            kwargs = {"return_entropy": True}
            # logits, entropies, soft_out, stds = model(img, **kwargs)
            logits, entropies = model(img)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            all_entropies.extend(entropies.cpu().numpy())
            # all_softs.extend(soft_out.cpu().numpy())
            # all_stds.extend(stds.cpu().numpy())
            print(f'\r calculating entropies for test {i}', end='')
            i +=1


        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()

    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies
                    #  "all_softs_test":  all_softs,
                    #  "all_std_test": all_stds,
                    #  "all_labels_train": all_labels_train,
                    #  "all_scores_train": all_scores_train,
                    #  "all_entropies_train": all_entropies_train
                     }
    run_name = config('run_name')
    labels_info_path = "Results/{run_name}"
    if not os.path.exists(labels_info_path):
        os.makedirs(labels_info_path)
    labels_info_path = os.path.join(labels_info_path, 'entropies.json')
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)



    print(f'Best model at epoch {best_epoch} has been saved')

    print('Saving the losses summary.....')
    losses_info = { "best_epoch": best_epoch,
                    "train_loss_best_epoch": train_loss_best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_accuracy,
                    "training_loss": loss_values_epoch,
                    "validation_loss": loss_values_val
                                                         }
    losses_summary_path = f"Model/losses/model_{j}_noise_std_{noise_std}_losses_summary.json"
    with open(losses_summary_path, 'w') as fp:
        json.dump(losses_info, fp, indent=2)

    return loss_values_epoch, loss_values_val, all_scores, all_labels



def evaluate_model(model, test_loader, text_inputs, device):

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            # img, text = img.cuda(), text.cuda()
            img_feats = model.encode_image(img)
            text_feats = model.encode_text(text_inputs)

            # img_feats /= img_feats.norm(dim=-1, keepdim=True)
            # text_feats /= text_feats.norm(dim=-1, keepdim=True)
            logits = torch.matmul(img_feats, text_feats.T)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
    return  all_scores, all_labels    

def evaluate_model_freeze(model, test_loader, device):
    

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            logits = model(img)


            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
    return  all_scores, all_labels    


def evaluate_model_freeze_adv(model, test_loader, device, eps):
    

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    for i, (images, labels) in enumerate(tqdm(test_loader)):
        img , text = images.to(device), labels.to(device)

        if config('attack_type').upper() == "FGSM":
            img_adv = fast_gradient_method(model, img, eps=eps, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, targeted=False)
        elif config('attack_type').upper() == "PGD":
            img_adv = projected_gradient_descent(model, img, eps, eps/4, 40, np.inf)

        img_adv = img_adv.cuda()
        with torch.no_grad():
            logits = model(img_adv)
            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            print(f'\r {i}', end='')
            i +=1

    # Convert the lists of scores and labels to NumPy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    return  all_scores, all_labels    



def evaluate_model_freeze_adv_lora(model, test_loader, device, eps):
    

    model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    for i, (images, labels) in enumerate(tqdm(test_loader)):
        img , text = images.to(device), labels.to(device)

        if config('attack_type').upper() == "FGSM":
            img_adv = fast_gradient_method(model.backbone_model, img, eps=eps, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, targeted=False)
        elif config('attack_type').upper() == "PGD":
            img_adv = projected_gradient_descent(model.backbone_model, img, eps, eps/4, 40, np.inf)

        img_adv = img_adv.cuda()
        with torch.no_grad():
            logits = model.backbone_model(img_adv)
            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  # Convert predicted tensor to numpy array and extend the list
            all_labels.extend(labels.numpy())  # Extend the list with true labels
            print(f'\r {i}', end='')
            i +=1

    # Convert the lists of scores and labels to NumPy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    return  all_scores, all_labels    


def evaluate_model_cifar_ensemble(ensemble, test_loader, text_inputs, device):

    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels = []
    i = 0
    with torch.no_grad():

        for images, labels in test_loader:
            img , text = images.to(device), labels.to(device)
            
            logits = []
            for model in ensemble:

                img_feats = model.encode_image(img)
                text_feats = model.encode_text(text_inputs)
                l = torch.matmul(img_feats, text_feats.T)

                sft = torch.softmax(l, 1)

                logits.append(sft)

            logits = torch.stack(logits).mean(0)

            predicted = torch.argmax(logits, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
    return  all_scores, all_labels 



def evaluate_model_cam_ensemble_freeze_norm(ensemble, test_loader, device):
    """Evalate for normal model (w/o lora)
    the only diff is w/o .backbone_model()
    """

    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    i = 0
    with torch.no_grad():
      
        for images, labels in test_loader:
            
            img , text = images.to(device), labels.to(device)
 
            logits = [] 
            softs, entropies = [],[]
            for model in ensemble:
 
                model = model.cuda()
                l = model(img)
                sft = torch.softmax(l, 1)

                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
                
            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            entropies = torch.stack(entropies).mean(0)

            predicted = torch.argmax(softs, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        # all_scores = np.array(all_scores)
        # all_labels = np.array(all_labels)

        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds
                                                }

    labels_info_path = f"Results/entropies_model_cam.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 


def evaluate_model_cam_ensemble_freeze_lora(ensemble, test_loader, device):

    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    i = 0
    with torch.no_grad():
      
        for images, labels in test_loader:
            
            img , text = images.to(device), labels.to(device)
 
            logits = [] 
            softs, entropies = [],[]
            for model in ensemble:
 
                model = model.cuda()
                l = model.backbone_model(img)
                sft = torch.softmax(l, 1)

                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
                
            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            entropies = torch.stack(entropies).mean(0)

            predicted = torch.argmax(softs, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

            print(f'\r {i}', end='')
            i +=1

        # Convert the lists of scores and labels to NumPy arrays
        # all_scores = np.array(all_scores)
        # all_labels = np.array(all_labels)

        all_scores = np.array(all_scores).tolist()
        all_labels = np.array(all_labels).tolist()
        all_entropies = np.array(all_entropies).tolist()
        all_softs = np.array(all_softs).tolist()
        all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds
                                                }

    labels_info_path = f"Results/entropies_model_cam.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

# write a wrapper to get ensemble in and produce the average results in forward pass
class EnsembleWrapperLoRa(nn.Module):
    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble

    def forward(self, x):
        logits = []
        for model in self.ensemble:
            model = model.cuda()
            logits.append(model.backbone_model(x))
        logits = torch.stack(logits).mean(0).cuda()
        return logits

class EnsembleWrapperNormal(nn.Module):
    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble

    def forward(self, x):
        logits = []
        for model in self.ensemble:
            model = model.cuda()
            logits.append(model(x))
        logits = torch.stack(logits).mean(0).cuda()
        return logits

    # def get_entropies(self, x):
    #     logits = [], entropies = []
    #     for model in self.ensemble:
    #         logits.append(model(x))
    #         E = -torch.softmax(logits[-1] + 1e-8, dim=1) * torch.log_softmax(logits[-1] + 1e-8, dim=1)
    #         entropies.append(E.sum(1))
    #     logits = torch.stack(logits).mean(0)
    #     entropies = torch.stack(entropies).mean(0)
    #     return logits, entropies


class LoRaWrapperNormal(nn.Module):
    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble

    def forward(self, x):
        logits = []
        for model in self.ensemble:
            model = model.cuda()
            logits.append(model.backbone_model(x))
        logits = torch.stack(logits).mean(0).cuda()
        return logits


from tqdm import tqdm
import csv
from utils import progress_bar

def evaluate_ensemble_confusion_adv_ensemble(ensemble, test_loader, device, eps, path_results):
    """ Evaluate the robustness or normal model
        the diff is to remove the backbone_model() from the model
    """
    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    ens = EnsembleWrapperNormal(ensemble)
    ens = ens.cuda()

    file_path = os.path.join(path_results, f"confusion_adv_{config('model_type')}_{config('attack_type')}.csv")
    f = open(file_path, 'w')
    writer = csv.writer(f)
    header = [
        'source/target', '0', '1', '2', '3', '4'
    ]

    writer.writerow(header)

    print(f"Attacking the network using method: {config('attack_type')} and eps: {eps}")

    for particle_id_0 in range(len(ensemble)):
        data = []
        correct = 0
        misclassified = 0
        test_loss = 0
        mis_examples = []
        benign_labels = []
        net_sample = ensemble[particle_id_0]
        net_sample = net_sample.cuda()
        total = 0

        for i, (images, labels) in enumerate(tqdm(test_loader)):

            entropies, logits, stds, softs = [], [], [], []
                
            img , labels = images.to(device), labels.to(device)


            # generate adv from each particle

            if config('attack_type').upper() == 'FGSM':
                # fgsm
                images_adv = fast_gradient_method(net_sample, img, eps=eps, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, targeted=False)
            elif config('attack_type').upper() == 'PGD':
                # print("Attack using PGD method")
                images_adv = projected_gradient_descent(net_sample, img, eps=eps, eps_iter=eps/4, nb_iter=40, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, sanity_checks=False)

            images_adv = images_adv.to(device)  

            outputs = net_sample(images_adv)
            _, predicted = torch.max(outputs.data,
                                     1)  # Prediction on the clean image
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            misclassified += (predicted != labels).sum().item()

            # get the misclassified examples
            mis = (predicted != labels)
            mis_examples.append(images_adv[mis])
            benign_labels.append(labels[mis])

            progress_bar(
                i,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                    test_loss / (i + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
        print( 
            "Accuracy of the model on adv test images of particle {} is : {} ({}/{}) %"
            .format(particle_id_0, 100 * correct / total, correct, total))

        postfix = "({}/{})".format(correct, total)
        data.append(str(particle_id_0) + postfix)
        for particle_id in range(len(ensemble)):
            total = 0
            correct = 0
            misclassified = 0
            net_sample = ensemble[particle_id]
            net_sample = net_sample.cuda()
            for j in range(len(mis_examples)):
                images = mis_examples[j]
                labels = benign_labels[j]
                outputs = net_sample(images)
                _, predicted = torch.max(outputs.data,
                                         1)  # Prediction on the clean image
                total += images.size(0)
                misclassified += (predicted != labels).sum().item()
                correct += (predicted == labels).sum().item()

            print(
                "Robustness of the model {} on adv images of particle {} is : {:.1f} ({}/{}) %"
                .format(particle_id, particle_id_0, 100 * correct / total,
                        correct, total))
            robustness_float = 100.0 * correct / total
            robustness_str = "{:.1f}".format(robustness_float)
            data.append(robustness_str)
        writer.writerow(data)
    f.close()




def evaluate_ensemble_confusion_adv_lora(ensemble, test_loader, device, eps, path_results):
    """ Evaluate the robustness or normal model
        the diff is to remove the backbone_model() from the model
    """
    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    ens = LoRaWrapperNormal(ensemble)
    ens = ens.cuda()

    file_path = os.path.join(path_results, f"confusion_adv_{config('model_type')}_{config('attack_type')}.csv")
    f = open(file_path, 'w')
    writer = csv.writer(f)
    header = [
        'source/target', '0', '1', '2', '3', '4'
    ]

    writer.writerow(header)

    print(f"Attacking the network using method: {config('attack_type')} and eps: {eps}")

    for particle_id_0 in range(len(ensemble)):
        data = []
        correct = 0
        misclassified = 0
        test_loss = 0
        mis_examples = []
        benign_labels = []
        net_sample = ensemble[particle_id_0].backbone_model
        net_sample = net_sample.cuda()
        total = 0

        for i, (images, labels) in enumerate(tqdm(test_loader)):

            entropies, logits, stds, softs = [], [], [], []
                
            img , labels = images.to(device), labels.to(device)


            # generate adv from each particle

            if config('attack_type').upper() == 'FGSM':
                # fgsm
                images_adv = fast_gradient_method(net_sample, img, eps=eps, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, targeted=False)
            elif config('attack_type').upper() == 'PGD':
                # print("Attack using PGD method")
                images_adv = projected_gradient_descent(net_sample, img, eps=eps, eps_iter=eps/4, nb_iter=40, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, sanity_checks=False)

            images_adv = images_adv.to(device)  

            outputs = net_sample(images_adv)
            _, predicted = torch.max(outputs.data,
                                     1)  # Prediction on the clean image
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            misclassified += (predicted != labels).sum().item()

            # get the misclassified examples
            mis = (predicted != labels)
            mis_examples.append(images_adv[mis])
            benign_labels.append(labels[mis])

            progress_bar(
                i,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                    test_loss / (i + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
        print( 
            "Accuracy of the model on adv test images of particle {} is : {} ({}/{}) %"
            .format(particle_id_0, 100 * correct / total, correct, total))

        postfix = "({}/{})".format(correct, total)
        data.append(str(particle_id_0) + postfix)
        for particle_id in range(len(ensemble)):
            total = 0
            correct = 0
            misclassified = 0
            net_sample = ensemble[particle_id].backbone_model
            net_sample = net_sample.cuda()
            for j in range(len(mis_examples)):
                images = mis_examples[j]
                labels = benign_labels[j]
                outputs = net_sample(images)
                _, predicted = torch.max(outputs.data,
                                         1)  # Prediction on the clean image
                total += images.size(0)
                misclassified += (predicted != labels).sum().item()
                correct += (predicted == labels).sum().item()

            print(
                "Robustness of the model {} on adv images of particle {} is : {:.1f} ({}/{}) %"
                .format(particle_id, particle_id_0, 100 * correct / total,
                        correct, total))
            robustness_float = 100.0 * correct / total
            robustness_str = "{:.1f}".format(robustness_float)
            data.append(robustness_str)
        writer.writerow(data)
    f.close()



def evaluate_model_cam_ensemble_freeze_adv_norm(ensemble, test_loader, device, eps):
    """ Evaluate the robustness or normal model
        the diff is to remove the backbone_model() from the model
    """
    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    ens = EnsembleWrapperNormal(ensemble)
    ens = ens.cuda()
    # with torch.no_grad():
    # apply tqdm to test_loader


    print("Attacking the network using method: {} and eps: {}".format(config('attack_type'), eps))
    for i, (images, labels) in (enumerate(tqdm(test_loader))):

        entropies, logits, stds, softs = [], [], [], []
            
        img , text = images.to(device), labels.to(device)


        if config('attack_type').upper() == 'FGSM':
            # print("Attack using FGSM method")
            # fgsm
            img_adv = fast_gradient_method(ens, img, eps=eps, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, targeted=False)
        elif config('attack_type').upper() == 'PGD':
            # print("Attack using PGD method")
            img_adv = projected_gradient_descent(ens, img, eps=eps, eps_iter=eps/4, nb_iter=40, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, sanity_checks=False)

        img_adv = img_adv.to(device)

        with torch.no_grad():

            for model in ensemble:

                # remove the backbone for normal model
                l = model(img_adv)
                sft = torch.softmax(l, 1)
                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
                
            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            entropies = torch.stack(entropies).mean(0)

            predicted = torch.argmax(softs, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

    # Convert the lists of scores and labels to NumPy arrays
    # all_scores = np.array(all_scores)
    # all_labels = np.array(all_labels)

    all_scores = np.array(all_scores).tolist()
    all_labels = np.array(all_labels).tolist()
    all_entropies = np.array(all_entropies).tolist()
    all_softs = np.array(all_softs).tolist()
    all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                    "all_scores_test": all_scores,   
                    "all_entropies_test":  all_entropies,
                    "all_softs_test":  all_softs,
                    "all_std_test": all_stds
                                                }

    labels_info_path = f"Results/entropies_model_cam.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 



def evaluate_model_cam_ensemble_freeze_adv_lora(ensemble, test_loader, device, eps):
    """ Evaluate the robustness or normal model
        the diff is to remove the backbone_model() from the model
    """
    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    ens = EnsembleWrapperLoRa(ensemble)
    ens = ens.cuda()

    print("Evaluate the LORA network using method: {} and eps: {}".format(config('attack_type'), eps))
    for i, (images, labels) in (enumerate(tqdm(test_loader))):

        entropies, logits, stds, softs = [], [], [], []
            
        img , text = images.to(device), labels.to(device)


        if config('attack_type').upper() == 'FGSM':
            # FGSM
            img_adv = fast_gradient_method(ens, img, eps=eps, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, targeted=False)
        elif config('attack_type').upper() == 'PGD':
            # PGD 
            img_adv = projected_gradient_descent(ens, img, eps=eps, eps_iter=eps/4, nb_iter=20, norm=np.inf, clip_min=0.0, clip_max=1.0, y=None, sanity_checks=False)
        
        img_adv = img_adv.to(device)
 
        with torch.no_grad():

            for model in ensemble:
 
                # remove the backbone for normal model
                l = model.backbone_model(img_adv)
                sft = torch.softmax(l, 1)
                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
                
            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            entropies = torch.stack(entropies).mean(0)

            predicted = torch.argmax(softs, dim=1)
            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

    # Convert the lists of scores and labels to NumPy arrays
    # all_scores = np.array(all_scores)
    # all_labels = np.array(all_labels)

    all_scores = np.array(all_scores).tolist()
    all_labels = np.array(all_labels).tolist()
    all_entropies = np.array(all_entropies).tolist()
    all_softs = np.array(all_softs).tolist()
    all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds
                                                }

    labels_info_path = f"Results/entropies_model_cam.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 


def evaluate_model_uncertainty_lora(ensemble, test_loader, device):
    """ Evaluate the uncertainty or lora model
    """
    # model.eval()  # Set the model to evaluation mode

    # Evaluation loop
    all_scores = []
    all_labels, all_entropies, all_softs, all_stds = [], [], [], []
    ens = EnsembleWrapperLoRa(ensemble)
    ens = ens.cuda()


    for i, (images, labels) in (enumerate(tqdm(test_loader))):

        entropies, logits, stds, softs = [], [], [], []
        misclassified_imgs, misclassified_labels = [], []
            
        img , text = images.to(device), labels.to(device)

        with torch.no_grad():

            for model in ensemble:
 
                # remove the backbone for normal model
                l = model.backbone_model(img)
                sft = torch.softmax(l, 1)
                entropies.append((-sft * torch.log(sft + 1e-8)).sum(1))
                logits.append(l)
                softs.append(sft)
                
            logits = torch.stack(logits).mean(0)
            stds = torch.stack(softs).std(0)
            softs = torch.stack(softs).mean(0)
            entropies = torch.stack(entropies).mean(0)

            predicted = torch.argmax(softs, dim=1)

            # get misclassified samples
            misclassified = predicted != labels
            # get imgs and labels of misclassified samples
            mis_img = img[misclassified]
            mis_labels = labels[misclassified]
            # add misclassified samples to the list
            misclassified_imgs.extend(mis_img.cpu().numpy())
            misclassified_labels.extend(mis_labels.cpu().numpy())







            all_scores.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.numpy())  
            all_entropies.extend(entropies.cpu().numpy())
            all_softs.extend(softs.cpu().numpy())
            all_stds.extend(stds.cpu().numpy())            

    # Convert the lists of scores and labels to NumPy arrays
    # all_scores = np.array(all_scores)
    # all_labels = np.array(all_labels)

    all_scores = np.array(all_scores).tolist()
    all_labels = np.array(all_labels).tolist()
    all_entropies = np.array(all_entropies).tolist()
    all_softs = np.array(all_softs).tolist()
    all_stds = np.array(all_stds).tolist()


    labels_info = {  "all_labels_test": all_labels,
                     "all_scores_test": all_scores,   
                     "all_entropies_test":  all_entropies,
                     "all_softs_test":  all_softs,
                     "all_std_test": all_stds
                                                }

    labels_info_path = f"Results/entropies_model_cam.json"
    with open(labels_info_path, 'w') as fp:
        json.dump(labels_info, fp, indent=2)

    return  all_scores, all_labels 











