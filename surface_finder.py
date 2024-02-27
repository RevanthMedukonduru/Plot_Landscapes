import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import utils

from bigmodelvis import Visualization

parser = argparse.ArgumentParser(description='Computes values for plane visualization')
parser.add_argument('--dir', type=str, default='/tmp/plane', metavar='DIR',
                    help='training directory (default: /tmp/plane)')

parser.add_argument('--margin_left', type=float, default=0.2, metavar='M', #0.9
                    help='left margin (default: 0.2)')
parser.add_argument('--margin_right', type=float, default=0.1, metavar='M', #0.3
                    help='right margin (default: 0.2)')
parser.add_argument('--margin_bottom', type=float, default=0.2, metavar='M', #0.4
                    help='bottom margin (default: 0.)')
parser.add_argument('--margin_top', type=float, default=0.1, metavar='M', #0.3
                    help='top margin (default: 0.2)')
parser.add_argument('--num_points', type=int, default=40, metavar='N',
                    help='number of points between models (default: 6)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, action='append', metavar='CKPT', required=True,
                    help='checkpoint to eval, pass all the models through this parameter')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument("--opt", type=int, default=1, help="No of Lora particles")
args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

checkpoints = args.ckpt[0].split(',')
args.ckpt = args.ckpt[0].split(',')
print("args.ckpt: ", checkpoints)


torch.backends.cudnn.benchmark = True

architecture = getattr(models, args.model)
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    architecture.preprocess,
    args.use_test,
    shuffle_train=False
)
base_model = architecture.base
base_model.cuda()

criterion = F.cross_entropy
regularizer = utils.l2_regularizer(args.wd)

def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

def get_weights(model):
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.delta_models[0].parameters()])

G = args.num_points
L = (len(args.ckpt))

# Load the weights into model, and get the weights
temp_params = list()    
for checkpoint_idx in range(L):
    path = checkpoints[checkpoint_idx]
    print(path)
    temp_checkpoint = torch.load(path)
    for name, param in base_model.delta_models[0].named_parameters():
        param.data.copy_(temp_checkpoint[name])
    temp_params.append(get_weights(base_model))

# Calculating unit vector u in direction of Model 0 to Model 2
u = temp_params[2] - temp_params[0] # Alpha
dx = np.linalg.norm(u)
u /= dx
print("U, DX: ",u.shape, dx)

# Calculating unit vector v in direction of Model 0 to Model 1
v = temp_params[1] - temp_params[0] # Beta
dy = np.linalg.norm(v)
v /= dy
print("V, DY: ",v.shape, dy)

# Calculating bend coordinates for each model
bend_coordinates = list()
for i in range(L):
    bend_coordinates.append(get_xy(temp_params[i], temp_params[0], u, v))
    print(f"BEND COORDINATES for {i}: ", bend_coordinates[i])

ts = np.linspace(0.0, 1.0, G)
alphas = np.linspace(0.0-args.margin_left, 1.0+args.margin_right, G)
betas = np.linspace(0.0-args.margin_bottom, 1.0+args.margin_top, G)

if 0.0 not in alphas or 1.0 not in alphas or 0.0 not in betas or 1.0 not in betas: # By default this setting will not cause error, but if margins changed.
    if 0.0 not in alphas:
        alphas = np.append(0.0, alphas)
    if 1.0 not in alphas:
        alphas = np.append(alphas, 1.0)
    if 0.0 not in betas:
        betas = np.append(0.0, betas)
    if 1.0 not in betas:
        betas = np.append(betas, 1.0)
    print("WARNING: The margins are not set properly, So 0.0 and 1.0 is added to the alphas and betas, so total no of points are: ", len(alphas), len(betas))

G = len(alphas)

tr_loss = np.zeros((L, G, G))
tr_nll = np.zeros((L, G, G))
tr_acc = np.zeros((L, G, G))
tr_err = np.zeros((L, G, G))

te_loss = np.zeros((L, G, G))
te_nll = np.zeros((L, G, G))
te_acc = np.zeros((L, G, G))
te_err = np.zeros((L, G, G))

grid = np.zeros((L, G, G, 2))

base_model = architecture.base
base_model.cuda()

columns = ["I", 'X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

step=0
no_of_left_rotations=0
for i in range(1): # took model 0 as center/starting point and travelling across u, v unit vectors (across model 1, model 2)
    for iter_i, alpha in enumerate(alphas):
        for iter_j, beta in enumerate(betas):
            w = temp_params[0] + alpha * dx * u + beta * dy * v
            offset = 0
            for parameter in base_model.delta_models[0].parameters():
                size = np.prod(parameter.size())
                value = w[offset:offset+size].reshape(parameter.size())
                parameter.data.copy_(torch.from_numpy(value))
                offset += size

            utils.update_bn(loaders['train'], base_model)

            tr_res = utils.test(loaders['train'], base_model, criterion, regularizer)
            te_res = utils.test(loaders['test'], base_model, criterion, regularizer)
            
            tr_loss_v, tr_nll_v, tr_acc_v = tr_res['loss'], tr_res['nll'], tr_res['accuracy']
            te_loss_v, te_nll_v, te_acc_v = te_res['loss'], te_res['nll'], te_res['accuracy']
            
            grid[i, iter_i, iter_j] = [bend_coordinates[i][0]+(alpha*dx), bend_coordinates[i][1]+(beta*dy)]
            
            tr_loss[i, iter_i, iter_j] = tr_loss_v
            tr_nll[i, iter_i, iter_j] = tr_nll_v
            tr_acc[i, iter_i, iter_j] = tr_acc_v
            tr_err[i, iter_i, iter_j] = 100.0 - tr_acc[i, iter_i, iter_j]
            
            te_loss[i, iter_i, iter_j] = te_loss_v
            te_nll[i, iter_i, iter_j] = te_nll_v
            te_acc[i, iter_i, iter_j] = te_acc_v
            te_err[i, iter_i, iter_j] = 100.0 - te_acc[i, iter_i, iter_j]
            
            values = [
                i, grid[i, iter_i, iter_j, 0], grid[i, iter_i, iter_j, 1], tr_loss[i, iter_i, iter_j], tr_nll[i, iter_i, iter_j], tr_err[i, iter_i, iter_j],
                te_nll[i, iter_i, iter_j], te_err[i, iter_i, iter_j]
            ]
            
            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
            if step % 25 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)
            step += 1
        
np.savez(
    os.path.join(args.dir, 'plane.npz'),
    ts=ts,
    bend_coordinates=bend_coordinates,
    alphas=alphas,
    betas=(1-alphas),
    grid=grid,
    tr_loss=tr_loss,
    tr_acc=tr_acc,
    tr_nll=tr_nll,
    tr_err=tr_err,
    te_loss=te_loss,
    te_acc=te_acc,
    te_nll=te_nll,
    te_err=te_err
)
