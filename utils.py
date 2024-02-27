import numpy as np
import os
import torch
import torch.nn.functional as F

import curves


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda()
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))


################################################################

import os
import json
import torch
import copy
import numpy as np
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
from decouple import config
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

class Paths():
    def __init__(self, config):
        self.path_results = ''
        self.path_store_model = ''
        self.path_figure=''

    def create_path(self):
        """
        This function creates a path for saving the best models and results
        :param settings: settings of the project
        :returns:
            path_results: the path for saving results
            path_saved_models: the path for saving the trained models
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.path_results = dir_path + '/Results/' 
        self.path_data = dir_path + '/Data/'
        self.path_store_model = dir_path +'/' + config('model_path') 
        # self.path_store_model = dir_path +'/' + '/Models/'
        self.path_figure = self.path_results + '/Figures/'
        self.path_checkpoints = self.path_store_model + '/losses/'

        Path(self.path_results).mkdir(parents=True, exist_ok=True)
        Path(self.path_figure).mkdir(parents=True, exist_ok=True)
        Path(self.path_store_model).mkdir(parents=True, exist_ok=True)
        Path(self.path_data).mkdir(parents=True, exist_ok=True)
        Path(self.path_checkpoints).mkdir(parents=True, exist_ok=True)



def add_noise_to_parameters(model, noise_std):
    device = next(model.parameters()).device  
    for param in model.parameters():
        noise = torch.randn(param.size(), device=device) * noise_std  
        param.data.add_(noise)



def generate_and_save_plot(loss_values, val_loss_values, save_path):

    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()



def generate_results(all_scores, all_labels, noise, i, paths):

    accuracy = accuracy_score(all_labels, all_scores)
    print(f"Accuracy: {accuracy}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_scores)
    print("Confusion Matrix:")
    print(conf_matrix)


    # Calculate evaluation metrics
    f1 = f1_score(all_labels, all_scores, average='weighted')
    precision = precision_score(all_labels, all_scores, average='weighted')
    recall = recall_score(all_labels, all_scores, average='weighted')

    # Prepare data for the bar plot
    print(f"results for model with {noise} noise: f1-score: {f1}, precision: {precision}, recall: {recall}")
    fil_path = paths + f'evaluation_model_{i}_noise_{noise}_metrics.txt'  

    with open(fil_path, 'w') as file:
        file.write('F1 Score: {:.4f}\n'.format(f1))
        file.write('Precision: {:.4f}\n'.format(precision))
        file.write('Recall: {:.4f}\n'.format(recall))
        file.write('Accuracy: {:.4f}\n'.format(accuracy))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix noise: {noise}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

    # Save confusion matrix 
    plt.savefig(paths + f'noise_{noise}_confusion_matrix_model_{i}.png')
    plt.close()

    #np.save(paths + f'noise_{noise}_scores.npy', all_scores)  

def bar_plot_diff(data, noise, path):
    keys = list(data.keys())
    values = list(data.values())

    plt.bar(keys, values)
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title(f'Bar Plot noise: {noise}')
    plt.xticks(rotation=45)
    plt.savefig(path + f'model_noise_{noise}_barplot.png')
    plt.close()


def block_diff(model, model_new):
    my_dict = {}

    blocks_to_compare = ['visual', 'token_embedding', 'ln_final']

    for block_name in blocks_to_compare:
        pretrained_block = dict(model.named_children())[block_name]
        fine_tuned_block = dict(model_new.named_children())[block_name]
        
        param_names =[]
        param_diff_f = []
        for (pretrained_param_name, pretrained_param), (fine_tuned_param_name, fine_tuned_param) in zip(pretrained_block.named_parameters(), fine_tuned_block.named_parameters()):
            param_diff = torch.norm(pretrained_param - fine_tuned_param)  
            
            param_names.append(pretrained_param_name)
            pp = param_diff.detach().cpu().numpy().item()
            param_diff_f.append(pp)

        
        my_dict[block_name] = sum(param_diff_f)

    return      my_dict   


def generate_equal_subsets(dataset, num_splits):
    torch.manual_seed(42)

    labels = torch.tensor(dataset.targets)
    splitter = StratifiedShuffleSplit(n_splits=num_splits, test_size=None, random_state=42)
    subset_indices = list(splitter.split(torch.zeros_like(labels), labels))

    subsets = []
    for indices in subset_indices:
        subset = torch.utils.data.Subset(dataset, indices[0])
        subsets.append(subset)

    return subsets


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster



def generate_particles(model , num_ensemble):
    particles = []
    for i in range(num_ensemble):
            particles.append(copy.deepcopy(model))

    print(f'number of individual models: {len(particles)}')  
    
    return particles      

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster



import sys
import time

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f