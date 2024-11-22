import math
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from torchvision import datasets, transforms

import os
import argparse

from tqdm import tqdm
import numpy as np

from models.psa_modules.respsa import ResPSA
from models.utils import Affine, AvgPool, AffineForPSA
from models.quantization_helper import (
    setup_quantizer, setup_quant_params_from_percentile, set_quant_params, quant_forward, get_imax_scale,
    set_int_ops
)

from matplotlib import pyplot
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', default='~/icml_data', type=str)
parser.add_argument('--download-dataset', action='store_true', default=False)
parser.add_argument('--quantize', action='store_true', default=False)
parser.add_argument('--qat', action='store_true', default=False)
parser.add_argument('--input-bits', default=8, type=int)
parser.add_argument('--weight-bits', default=8, type=int)
parser.add_argument('--bias-bits', default=8, type=int)
parser.add_argument('--output-bits', default=8, type=int)
parser.add_argument('--hidden-factor', default=4., type=float)
parser.add_argument('--embed-dim', default=96, type=int)
parser.add_argument('--do-kd', action='store_true', default=False)
parser.add_argument('--kd-lr', default=1e-3, type=float)
parser.add_argument('--kd-wd', default=0.0, type=float)
parser.add_argument('--kd-epochs', default=100, type=int)
parser.add_argument('--kd-t', default=1.0, type=float)
parser.add_argument('--kd-alpha', default=0.8, type=float)
parser.add_argument('--teacher-load', default=None, type=str)
parser.add_argument('--teacher-resnet', action='store_true', default=False)
parser.add_argument('--teacher-nl-act', type=str, choices=['psa','bspline', 'base'], default='base')
parser.add_argument('--teacher-hidden-factor', default=4., type=float)
parser.add_argument('--student-resnet', action='store_true', default=False)
parser.add_argument('--student-nl-act', type=str, choices=['psa','bspline', 'base'], default='psa')
parser.add_argument('--student-hidden-factor', default=4., type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--test-batch-size', default=128, type=int)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--disable-tqdm', action='store_true', default=False)
parser.add_argument('--quant-load', default=None, type=str)
parser.add_argument('--save', default=None, type=str)
parser.add_argument('--load', default=None, type=str)
parser.add_argument('--dump-params', action='store_true', default=False)
parser.add_argument('--dump-dir', type=str, default='./')
parser.add_argument('--seed', type=int, default=2023)
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print(f'Using device: {device}')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# train_kwargs = {'batch_size': args.batch_size}
train_kwargs = {'batch_size': args.batch_size, 'drop_last': True}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616))
])

trainset = datasets.CIFAR10('~/icml_data', train=True, download=False,
    transform=test_transform)
testset = datasets.CIFAR10('~/icml_data', train=False, download=False,
    transform=test_transform)

train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
val_loader = torch.utils.data.DataLoader(valset,**test_kwargs)
test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)


# ===============================================================================================

if args.teacher_resnet:
    from models.resnet import resnet20 as TeacherModel
else:
    if args.teacher_nl_act == 'psa':
        from models.psa_resmlp import ResMLP as TeacherModel
    elif args.teacher_nl_act == 'bspline':
        from models.bspline_resmlp import ResMLP as TeacherModel
    elif args.teacher_nl_act == 'base':
        from models.resmlp import ResMLP as TeacherModel
    else:
        raise NotImplementedError

if args.student_resnet:
    from models.resnet import resnet20 as StudentModel
else:
    if args.student_nl_act == 'psa':
        from models.psa_resmlp import ResMLP as StudentModel
    elif args.student_nl_act == 'bspline':
        from models.bspline_resmlp import ResMLP as StudentModel
    elif args.teacher_nl_act == 'base':
        from models.resmlp import ResMLP as StudentModel
    else:
        raise NotImplementedError

# ===============================================================================================

model = StudentModel(hidden_factor=args.student_hidden_factor, embed_dim=args.embed_dim)
model = model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()


def test(test_model, loader):
    global best_acc
    test_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader, disable=args.disable_tqdm)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = test_model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print("Accuray: {:.2f}".format(correct/total*100.))

    return correct/total


if args.load is not None:
    print(f'Loading model from: {args.load}')
    model.load_state_dict(torch.load(args.load, weights_only=True))
    model = model.to(device)
    print(model)


print('\nBaseline Accuracy')
test(model, test_loader)

# exit(1)

print('=' * 100)
print('\nAbsorbing Layers')
model.absorb_affines()
model = model.to(device)

print('\n')
print(model)

print('\nPre Model Accuracy:')
test(model, test_loader)
# exit(1)

layers_to_quantize = [nn.Linear, ResPSA, Affine, AvgPool]
if args.quantize:
    print('\nQuantizing Layers:')
    print(layers_to_quantize)

    quant_config = {
        'input_bits': args.input_bits,
        'weight_bits': args.weight_bits,
        'bias_bits': args.bias_bits,
        'output_bits': args.output_bits,
    }

    print('Quantization Config:')
    pprint(quant_config)

    setup_quantizer(
        model, layers_to_quantize, train_loader, int(len(train_loader) * 0.2),
        quant_config,
        load_path=args.quant_load,
        find_activation_ranges=args.quant_load is None,
        do_int_ops=False,
        disable_tqdm=args.disable_tqdm,
        device=device
    )

    best_percentile = None
    if not args.quant_load:
        # print('\nFinding best percentile using a val_split on training dataset')
        percentiles_to_search = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 92, 95, 97, 100]
        # # percentiles_to_search = [80, 85, 90, 95, 97, 100]
        best_percentile = -1
        best_accuracy = 0
        for scale_percentile in percentiles_to_search:
            print(f'Percentile = {scale_percentile}')

            set_quant_params(model, layers_to_quantize, percentile=scale_percentile, reset_weight_scaler=True)
            val_acc = test(model, val_loader)
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_percentile = scale_percentile
        print(f'\nFound best percentile at: {best_percentile}')

    # best_percentile = 30
    set_quant_params(model, layers_to_quantize, percentile=best_percentile, reset_weight_scaler=True)

    print('\nInitial quantized model accuracy:')
    set_int_ops(model, layers_to_quantize, True)
    test(model, test_loader)


if args.qat:
    # Disable INT ops for QAT
    # other wait an year
    set_int_ops(model, layers_to_quantize, False)

    # Re-Create the full dataset
    train_transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616))
    ])
    trainset = datasets.CIFAR10('~/icml_data', train=True, download=False,
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)

    ce_loss = nn.CrossEntropyLoss()
    kd_optimizer = optim.AdamW(model.parameters(), lr=args.kd_lr, weight_decay=args.kd_wd)

    if args.do_kd:
        # teacher_model = TeacherModel()
        teacher_model = TeacherModel(hidden_factor=args.teacher_hidden_factor, embed_dim=args.embed_dim)
        teacher_model = teacher_model.to(device)

        assert args.teacher_load is not None
        print(f'Loading model: {args.teacher_load}')
        # teacher_model = torch.load(args.teacher_load).to(device)
        teacher_model.load_state_dict(torch.load(args.teacher_load, weights_only=True), strict=True)
        print('Teacher Model Accuracy:')
        test(teacher_model, test_loader)

    def kd_train(epoch):
        print(f'Epoch: {epoch}')
        if args.do_kd:
            teacher_model.eval()
        model.train()
        train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, disable=args.disable_tqdm)):
            inputs, labels = inputs.to(device), labels.to(device)
            kd_optimizer.zero_grad()

            student_logits = model(inputs)

            if args.do_kd:
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)

                # Calculate the soft targets loss.
                # Scaled by T**2 as suggested by the authors of the paper
                # "Distilling the knowledge in a neural network"
                soft_targets = nn.functional.softmax(teacher_logits / args.kd_t, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / args.kd_t, dim=-1)
                soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (args.kd_t**2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)
                loss = soft_targets_loss

                # Weighted sum of the two losses
                loss = (args.kd_alpha * soft_targets_loss) + ((1 - args.kd_alpha) * label_loss)
            else:
                loss = ce_loss(student_logits, labels)

            loss.backward()
            kd_optimizer.step()

            train_loss += loss.item()

        print('[{}/{}] Train Loss = {:.3f}'.format(
            epoch, args.kd_epochs, train_loss))

    kd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        kd_optimizer, args.kd_epochs, eta_min=1e-5)

    for epoch in range(args.kd_epochs):
        kd_train(epoch)
        kd_scheduler.step()
        set_quant_params(model, layers_to_quantize, reset_weight_scaler=False)
        test(model, test_loader)

    set_quant_params(model, layers_to_quantize, reset_weight_scaler=False)
    set_int_ops(model, layers_to_quantize, True)
    print('\nFinal quantized model:')
    test(model, test_loader)

    if args.save is not None:
        torch.save(model.state_dict(), f'{args.save}')


if args.dump_params:
    param_dict = model.dump_params(quantized=True)
    file_path = os.path.join(args.dump_dir, 'quant_params.pkl')
    with open(file_path, 'wb') as fp:
        pickle.dump(param_dict, fp)

    input, labels = next(train_loader._get_iterator())
    input = input.to(device)
    labels = labels.to(device)
    in_f = model.patch_gen(input)
    n = len(model.layers) + 1
    inter = [None] * n
    model.eval()
    with torch.no_grad():
        for i in range(n):
            y = model(input, early_exit_after_block=i)
            if i < n - 1:
                y_abs_max = model.layers[i].layers[0].affine_1.input_abs_max
                y_bits = model.layers[i].layers[0].affine_1.input_bits
            else:
                y_abs_max = model.classifier.input_abs_max
                y_bits = model.classifier.input_bits
            y, _ = quant_forward(y, y_abs_max, y_bits, dequantize=False)
            inter[i] = y

        output = model(input)
        output_abs_max = model.classifier.output_abs_max
        output_bits = model.classifier.output_bits
        output, _ = quant_forward(output, output_abs_max, output_bits, dequantize=False)


    input_abs_max = model.embed_linear.input_abs_max
    input_bits = model.embed_linear.input_bits
    input, _ = quant_forward(model.patch_gen(input), input_abs_max, input_bits, dequantize=False)

    inter = [ t.cpu().detach().numpy() for t in inter ]
    input = input.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    samples_dict = {}
    samples_dict['input'] = input
    samples_dict['labels'] = labels
    samples_dict['inter'] = inter
    samples_dict['output'] = output

    file_path = os.path.join(args.dump_dir, 'quant_samples.pkl')
    with open(file_path, 'wb') as fp:
        pickle.dump(samples_dict, fp)

    all_test_samples_dict = {}
    all_test_samples_dict['input'] = []
    all_test_samples_dict['labels'] = []
    all_test_samples_dict['output'] = []
    for batch_idx, (input, labels) in enumerate(tqdm(test_loader, disable=args.disable_tqdm)):
        input = input.to(device)
        labels = labels.to(device)
        output = model(input)

        input_abs_max = model.embed_linear.input_abs_max
        input_bits = model.embed_linear.input_bits
        input, _ = quant_forward(model.patch_gen(input), input_abs_max, input_bits, dequantize=False)

        output_abs_max = model.classifier.output_abs_max
        output_bits = model.classifier.output_bits
        output, _ = quant_forward(output, output_abs_max, output_bits, dequantize=False)

        input = input.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        all_test_samples_dict['input'].append(input)
        all_test_samples_dict['labels'].append(labels)
        all_test_samples_dict['output'].append(output)

    file_path = os.path.join(args.dump_dir, 'all_test_quant_samples.pkl')
    with open(file_path, 'wb') as fp:
        pickle.dump(all_test_samples_dict, fp)
