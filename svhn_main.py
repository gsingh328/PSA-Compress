import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

import argparse

from tqdm import tqdm
import numpy as np

from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', default='~/icml_data', type=str)
parser.add_argument('--download-dataset', action='store_true', default=False)
parser.add_argument('--do-kt', action='store_true', default=False)
parser.add_argument('--kt-lr', default=1e-3, type=float)
parser.add_argument('--kt-wd', default=0.0, type=float)
parser.add_argument('--kt-epochs', default=50, type=int)
parser.add_argument('--do-kd', action='store_true', default=False)
parser.add_argument('--kd-lr', default=1e-3, type=float)
parser.add_argument('--kd-wd', default=0.0, type=float)
parser.add_argument('--kd-epochs', default=300, type=int)
parser.add_argument('--kd-t', default=1.0, type=float)
parser.add_argument('--kd-alpha', default=0.8, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--test-batch-size', default=128, type=int)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--disable-tqdm', action='store_true', default=False)
parser.add_argument('--kt-save', default=None, type=str)
parser.add_argument('--kt-load', default=None, type=str)
parser.add_argument('--teacher-load', default=None, type=str)
parser.add_argument('--teacher-resnet', action='store_true', default=False)
parser.add_argument('--teacher-nl-act', type=str, choices=['psa','bspline', 'base'], default='base')
parser.add_argument('--teacher-hidden-factor', default=4., type=float)
parser.add_argument('--student-resnet', action='store_true', default=False)
parser.add_argument('--student-nl-act', type=str, choices=['psa','bspline', 'base'], default='psa')
parser.add_argument('--student-hidden-factor', default=4., type=float)
parser.add_argument('--bspline-lambda', default=1e-5, type=float)
parser.add_argument('--save', default=None, type=str)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nondeter', action='store_true', default=False)
args = parser.parse_args()

print('\nArgs:')
pprint(vars(args))

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print(f'\nUsing device: {device}')

print(f'Setting python, numpy and python seeds to = {args.seed}')
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not args.nondeter:
    print('Disabling any non-deterministic algorithms')
    print('You may want to enable "--nondeter" flag if max performance is required')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True


# train_kwargs = {'batch_size': args.batch_size}
train_kwargs = {'batch_size': args.batch_size, 'drop_last': True}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

train_transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4376821, 0.4437697, 0.47280442),
        (0.19803012, 0.20101562, 0.19703614))
])

print('\nTraining Transforms:')
print(train_transform)

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4376821, 0.4437697, 0.47280442),
        (0.19803012, 0.20101562, 0.19703614))
])

print('\nTest Transforms:')
print(test_transform)

trainset = datasets.SVHN('~/icml_data', split="train", download=False,
    transform=train_transform)
testset = datasets.SVHN('~/icml_data', split="test", download=False,
    transform=test_transform)

train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
criterion = nn.CrossEntropyLoss()

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

if args.teacher_resnet:
    teacher_model = TeacherModel()
else:
    teacher_model = TeacherModel(hidden_factor=args.teacher_hidden_factor, embed_dim=64, nlayers=2)
teacher_model = teacher_model.to(device)

print('\n\n' + '=' * 100)
print('Teacher Model')
print('=' * 100)
print(teacher_model)


def test(model, epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, disable=args.disable_tqdm)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print("[{}] Loss: {:.2f}".format(epoch, test_loss))
        print("[{}] Accuray: {:.2f}".format(epoch, correct/total*100.))


if args.teacher_load is not None:
    print(f'Loading model: {args.teacher_load}')
    if args.teacher_resnet:
        teacher_model = torch.load(args.teacher_load).to(device)
    else:
        teacher_model.load_state_dict(torch.load(args.teacher_load), strict=True)
    test(teacher_model, 0)
elif args.do_kd or args.do_kt:
    print('No teacher model found, but doing kd or kt!')
    exit(1)


student_model = StudentModel(hidden_factor=args.student_hidden_factor, embed_dim=64, nlayers=2)
student_model = student_model.to(device)

print('\n\n' + '=' * 100)
print('Student Model')
print('=' * 100)
print(student_model)

def freeze_student_for_kt(block_idx_to_freeze_upto):
    student_model.embed_linear.weight.requires_grad = False
    student_model.embed_linear.bias.requires_grad = False
    for block_idx in range(block_idx_to_freeze_upto):
        for param in student_model.layers[block_idx].parameters():
            param.requires_grad = False


def unfreeze_student():
    student_model.embed_linear.weight.requires_grad = True
    student_model.embed_linear.bias.requires_grad = True
    for param in student_model.layers.parameters():
        param.requires_grad = True

lmbda = args.bspline_lambda # regularization weight
lipschitz = False # lipschitz control

# Knowledge Transfer
if args.do_kt:
    print('\n\n' + '='*100)
    print('Knowledge Transfer')
    print('=' * 100)
    if args.kt_load is not None:
        print(f'Loading Knowledge-Transfered model: {args.kt_load}')
        student_model = torch.load(args.kt_load)
    else:
        # Copy Embedding
        with torch.no_grad():
            student_model.embed_linear.weight.copy_(teacher_model.embed_linear.weight)
            student_model.embed_linear.bias.copy_(teacher_model.embed_linear.bias)

        kt_criterion = nn.MSELoss()
        # kt_criterion = nn.CosineEmbeddingLoss()
        if args.student_nl_act != 'bspline':
            kt_optimizer = optim.AdamW(student_model.parameters(), lr=args.kt_lr, weight_decay=args.kt_wd)
        else:
            kt_optimizer = optim.AdamW(student_model.parameters_no_deepspline(), lr=args.kt_lr, weight_decay=args.kt_wd)
            kt_aux_optimizer = optim.Adam(student_model.parameters_deepspline())

        def kt_train(block_idx, epoch):
            print(f'Epoch: {epoch}')
            teacher_model.eval()
            student_model.train()
            freeze_student_for_kt(block_idx)
            train_loss = 0
            for batch_idx, (inputs, _) in enumerate(tqdm(train_loader, disable=args.disable_tqdm)):
                inputs = inputs.to(device)
                kt_optimizer.zero_grad()
                if args.student_nl_act == 'bspline':
                    kt_aux_optimizer.zero_grad()

                with torch.no_grad():
                    teacher_hidden_repr = teacher_model(inputs, early_exit_after_block=block_idx+1)
                student_hidden_repr = student_model(inputs, early_exit_after_block=block_idx+1)

                loss = kt_criterion(student_hidden_repr, teacher_hidden_repr)
                # b = inputs.shape[0]
                # loss = kt_criterion(student_hidden_repr.view(b, -1), teacher_hidden_repr.view(b, -1),
                #     target=torch.ones(b).to(device))

                # add regularization loss
                if args.student_nl_act == 'bspline':
                    if lipschitz is True:
                        loss = loss + lmbda * student_model.BV2()
                    else:
                        loss = loss + lmbda * student_model.TV2()

                loss.backward()
                kt_optimizer.step()
                if args.student_nl_act == 'bspline':
                    kt_aux_optimizer.step()

                train_loss += loss.item()

            print('[{}/{}] Train Loss = {:.3f}'.format(
                epoch, args.kt_epochs, train_loss))


        def kt_test(block_idx, epoch):
            teacher_model.eval()
            student_model.eval()
            freeze_student_for_kt(block_idx)
            eval_loss = 0
            for batch_idx, (inputs, _) in enumerate(tqdm(test_loader, disable=args.disable_tqdm)):
                inputs = inputs.to(device)

                with torch.no_grad():
                    teacher_hidden_repr = teacher_model(inputs, early_exit_after_block=block_idx+1)
                    student_hidden_repr = student_model(inputs, early_exit_after_block=block_idx+1)

                loss = kt_criterion(student_hidden_repr, teacher_hidden_repr)

                eval_loss += loss.item()

            print('[{}] Eval Loss = {:.3f}'.format(epoch, eval_loss))


        blocks_num = len(student_model.layers)
        print(f'No. of blocks to Knowledge Transfer = {blocks_num}')
        for block_idx in range(blocks_num):
            for epoch in range(args.kt_epochs):
                kt_train(block_idx, epoch)
                kt_test(block_idx, epoch)


    if args.kt_save is not None:
        torch.save(student_model, f'{args.kt_save}')


# Knowledge Distillation
if args.do_kd:
    print('\n\n' + '='*100)
    print('Knowledge Distillation')
    print('=' * 100)
else:
    print('\n\n' + '='*100)
    print('Simple Training')
    print('=' * 100)

ce_loss = nn.CrossEntropyLoss()

if args.student_nl_act != 'bspline':
    kd_optimizer = optim.AdamW(student_model.parameters(), lr=args.kd_lr, weight_decay=args.kd_wd)
else:
    kd_optimizer = optim.AdamW(student_model.parameters_no_deepspline(), lr=args.kd_lr, weight_decay=args.kd_wd)
    kd_aux_optimizer = optim.Adam(student_model.parameters_deepspline())

def kd_train(epoch):
    print(f'Epoch: {epoch}')
    if args.do_kd:
        teacher_model.eval()
    student_model.train()
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, disable=args.disable_tqdm)):
        inputs, labels = inputs.to(device), labels.to(device)
        kd_optimizer.zero_grad()
        if args.student_nl_act == 'bspline':
            kd_aux_optimizer.zero_grad()

        student_logits = student_model(inputs)

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
            if args.student_nl_act == 'bspline':
                 # add regularization loss
                if lipschitz is True:
                    loss = loss + lmbda * student_model.BV2()
                else:
                    loss = loss + lmbda * student_model.TV2()
        else:
            loss = ce_loss(student_logits, labels)
            if args.student_nl_act == 'bspline':
                 # add regularization loss
                if lipschitz is True:
                    loss = loss + lmbda * student_model.BV2()
                else:
                    loss = loss + lmbda * student_model.TV2()

        loss.backward()
        kd_optimizer.step()
        if args.student_nl_act == 'bspline':
            kd_aux_optimizer.step()

        train_loss += loss.item()

    print('[{}/{}] Train Loss = {:.3f}'.format(
        epoch, args.kd_epochs, train_loss))

if args.do_kd:
    unfreeze_student()

kd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        kd_optimizer, args.kd_epochs, eta_min=1e-5)
if args.student_nl_act == 'bspline':
    kd_aux_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        kd_aux_optimizer, args.kd_epochs, eta_min=1e-5)

for epoch in range(args.kd_epochs):
    kd_train(epoch)
    kd_scheduler.step()
    if args.student_nl_act == 'bspline':
        kd_aux_scheduler.step()
    test(student_model, epoch)

if args.save is not None:
    if args.student_resnet:
        torch.save(student_model, f'{args.save}')
    else:
        torch.save(student_model.state_dict(), f'{args.save}')
