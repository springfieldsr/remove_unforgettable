import collections

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
import re
import random
import string
import os
from torch.utils.data import DataLoader, Subset
from const import PATIENCE_EPOCH

from sklearn.datasets import fetch_20newsgroups


class CustomDataset:
    def __init__(self, dataset_name, root=None, transform=None, train=True, download=True, data=None):
        """
        Input:
        dataset_name
            string of desired dataset name
        root
            string of data download destination
        train
            bool to specify wheter the dataset is for train or test
        download
            bool to specify whether to download the dataset specified by dataset_name
        transform
            operations to perform transformation on the dataset
        data
            solely used during check interval
        """
        if dataset_name == "CIFAR10":
            self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        elif dataset_name == "CIFAR100":
            self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
        elif dataset_name == "20news":
            self.dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))

            alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", '', x)
            punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x.lower())

            texts = [alphanumeric(punc_lower(text)) for text in self.dataset.data]
            self.dataset = list(zip(texts, self.dataset.target))
        elif dataset_name == "check":
            self.dataset = data
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        return sample, label

    def __len__(self):
        return len(self.dataset)


def eval(model, test_dataloader, device):
    """
    Input:
    model
        pytorch model object to eval
    test_dataloader
        pytorch dataloader
    device
        string of either 'cuda' or 'cpu'

    Return:
        float accuracy on the validation set
    """
    model.eval()

    match_count, total_count = 0, 0
    for (X, y) in test_dataloader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            logits = model(X)
            match_count += torch.sum(torch.argmax(logits, dim=1) == y)
            total_count += len(X)

    return match_count / total_count


def train(model, epoch, train_dataset, test_dataloader, device, args):
    """
    Input:
    model
        pytorch model object to train
    epoch
        int number of total training epochs
    train_dataset
        dataset object which satisfies pytorch dataset format
    test_dataloader
        pytorch dataloader
    device
        string of either 'cuda' or 'cpu'
    """

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val_acc = 0
    patience = 0

    # indices of examples that have been memorized by the model in a streak
    memorized_indices = set()
    # counter to record memory streak
    memory_streak_counter = collections.defaultdict(int)

    for e in range(epoch):
        model.train()

        train_loss = 0
        batch_size = args.batch_size

        # if in need of memorized fraction, we do not remove any data points from training process
        # else, we remove memorized ones
        if args.return_memorized_fraction:
            feed_indices = torch.randperm(len(train_dataset)).tolist()
        else:
            feed_indices = [i for i in torch.randperm(len(train_dataset)).tolist() if i not in memorized_indices]
        shuffled_dataset = Subset(train_dataset, feed_indices)
        dataloader = DataLoader(shuffled_dataset, batch_size, shuffle=False)

        with tqdm.tqdm(total=len(dataloader) * batch_size, unit='it', unit_scale=True, unit_divisor=1000) as pbar:
            sum_acc = 0
            total = 0
            for X, y in dataloader:
                if X.shape[0]==1: continue # avoid batch size = 1
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)

                # if we are not evaluating baseline
                # then begin recording the memory streak
                # remove examples from training dataset if they met the streak
                if not args.baseline:
                    match = logits.argmax(dim=1) == y
                    batch_indices = feed_indices[total: total + X.shape[0]]
                    for idx, i in enumerate(batch_indices):
                        if memory_streak_counter[i] == -1: continue
                        if match[idx]:
                            memory_streak_counter[i] += 1
                            if memory_streak_counter[i] == args.streak:
                                memorized_indices.add(i)
                                memory_streak_counter[i] = 0
                        else:
                            # if we need fraction, set every wrongly predicted data point which has been correctly
                            # predicted before as -1
                            if args.return_memorized_fraction and memory_streak_counter[i] > 0:
                                memory_streak_counter[i] = -1
                            else:
                                memory_streak_counter[i] = 0

                acc = torch.sum(torch.argmax(logits, dim=1) == y).item()
                loss.backward()
                opt.step()

                # progress bar counter
                train_loss += loss.item()
                pbar.update(batch_size)
                sum_acc += acc
                total += X.shape[0]
                pbar.set_postfix(loss=loss.item(),
                                 acc=sum_acc / total)

        # TODO: feel free to remove this print
        if not args.baseline:
            print(f"Epoch {e} - size of memorized data {len(memorized_indices)}, size of training data {len(dataloader)} " )

        validation_accuracy = eval(model, test_dataloader, device)
        print("Epoch {} - Training loss: {:.4f}, Validation Accuracy: {:.4f}".format(e, train_loss / len(dataloader),
                                                                                     validation_accuracy))

        # if we met the check interval, then check examples that are supposedly memorized
        # if the model forgot about them, then move them back to the training dataset
        # TODO: in fraction computation, do we check the memorized or not?
        if e > args.streak and e % args.check_interval == 0 and not args.return_memorized_fraction:
            print('Entering check interval')
            memorized_indices_list = list(memorized_indices)
            memorized_dataset = CustomDataset("check", data=[train_dataset[i] for i in memorized_indices_list])
            memorized_dataloader = DataLoader(memorized_dataset, batch_size, shuffle=False)
            memorized_total = 0
            forgot_count = 0
            with torch.no_grad():
                for X, y in memorized_dataloader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X)
                    match = torch.argmax(logits, dim=1) == y #fix
                    batch_indices = memorized_indices_list[memorized_total: memorized_total + X.shape[0]]
                    for idx, i in enumerate(batch_indices):
                        if not match[idx]:
                            memorized_indices.remove(i)
                            forgot_count += 1
                    memorized_total += X.shape[0] #fix
            print("Model forgets {} examples out of {} total memorized ones".format(forgot_count,
                                                                                    len(memorized_indices_list)))

        # Early stopping
        if args.early_stop:
            if validation_accuracy > best_val_acc:
                best_val_acc = validation_accuracy
                patience = 0
            else:
                patience += 1
                if patience > PATIENCE_EPOCH:
                    print("Training early stops at epoch {}".format(e))
                    break

    if args.return_memorized_fraction:
        truly_unforgettable = [k for k, v in memory_streak_counter.items() if v > 0]
        return len(memorized_indices) / len(truly_unforgettable)

    return memorized_indices


def SaveEnvironment():
    pass


def GenerateEnvironment(args):
    seeds = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    result_dir = './results'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    expr_path = os.path.join(result_dir, seeds).replace("\\","/")
    args_path = os.path.join(expr_path, 'args.txt').replace("\\","/")
    os.mkdir(expr_path)
    print('Create enviornment at : {}'.format(expr_path))
    with open(args_path, 'w') as F:
        DumpOptionsToFile(args, F)
    return expr_path


def DumpOptionsToFile(args, fp):
    d = vars(args)
    for key,value in d.items():
        if type(value) == str:
            fp.write('{} = "{}"\n'.format(key, value))
        else:
            fp.write('{} = {}\n'.format(key, value))


def DumpIndicesToFile(noise, dataset_name, dir):
    dest = os.path.join(dir, dataset_name).replace("\\","/")
    with open(dest, 'w') as f:
        for n in noise:
            f.write(str(n) + '\n')
    return dest