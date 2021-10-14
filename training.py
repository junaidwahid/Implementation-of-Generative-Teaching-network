
from GTN.Models.learner import Learner
from GTN.Models.teacher import Teacher

from helper_functions import *
from dataloader import get_data_loaders
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parameters import *
from torch.autograd import grad

import higher as higher
# Set random seeds
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def train_and_nas(train_iterator,val_iterator, test_iterator):

    teacher = Teacher()
    params_to_train = list(teacher.parameters())

    # If we want to use a curriculum, we initialize the learnable parameters here
    if use_curriculum:
        curriculum = nn.Parameter(torch.randn(inner_loop_iterations, inner_loop_batch_size, noise_size), requires_grad=True)
        params_to_train += [curriculum]

    optimizer_teacher = optim.Adam(params_to_train, lr=learning_rate)

    # For each inner loop iterations, we use the same sequence of labels.
    # This allows the curriculum vectors to train to stable labels
    label = torch.tensor([x % num_classes for x in range(inner_loop_batch_size)])

    # For the inner loop loss, we use cross entropy
    loss_fn = nn.CrossEntropyLoss()

    # Here we initialize iterators on the train and val datasets

    for it, real_data in enumerate(train_loader):


        teacher.train()
        optimizer_teacher.zero_grad()

        # We also optimize the learner learning rate and momentum with the
        # outer loop updates
        learner_lr = teacher.learner_optim_params[0]
        learner_momentum = teacher.learner_optim_params[1]

        # Here we sample a learner with random number of conv filters
        learner = Learner()
        inner_optim = optim.SGD(learner.parameters(), lr=learner_lr.item(), momentum=learner_momentum.item())
        learner.train()

        inner_losses = []
        with higher.innerloop_ctx(learner, inner_optim, override={'lr': [learner_lr], 'momentum': [learner_momentum]}) as (flearner, diffopt):
            for step in range(inner_loop_iterations):

                # Data generation
                if use_curriculum:
                    z_vec = curriculum[step]
                else:
                    z_vec = torch.randn(inner_loop_batch_size, noise_size)

                one_hot = F.one_hot(label, num_classes)

                # Pass input to teacher to generate synthetic images
                teacher_output, teacher_target = teacher(z_vec, one_hot)

                # ====== Show intermediate generated images ======
                if step == 0:
                    print('------------------ Outer loop iteration', it + 1, '------------------')
                    print('Examples 0 - 9 from beginning of inner loop:')
                    background = Image.new('L', (img_size * imgs_per_row + imgs_per_row + 1, img_size + 2))
                    for i in range(imgs_per_row):  # indexes column
                        background.paste(generate_img(teacher_output[i],mnist_std, mnist_mean), (i * 28 + i + 1, 1))
                    background.show()

                if step == (inner_loop_iterations - 1):
                    print('Examples 0 - 9 from end of inner loop:')
                    background = Image.new('L', (img_size * imgs_per_row + imgs_per_row + 1, img_size + 2))
                    for i in range(imgs_per_row):  # indexes column
                        background.paste(generate_img(teacher_output[i],mnist_std, mnist_mean), (i * 28 + i + 1, 1))
                    background.show()

                # Pass teacher output to the learner
                learner_output = flearner(teacher_output)
                loss = loss_fn(learner_output, label)
                diffopt.step(loss)

                inner_losses.append(loss.item())

            correct = 0
            data, target = real_data
            output = flearner(data)
            loss = loss_fn(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy_train = correct / target.shape[0]

            print("Inner loop losses:", inner_losses)
            print("Train accuracy:", accuracy_train)

            # Compute accuracy on validation set
            data, target = next(val_iterator)
            output = flearner(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / outer_loop_batch_size
            print("Val accuracy:", accuracy)

            if (it == outer_loop_iterations - 1):
                # Compute accuracy on test set
                correct = 0
                for i, (data, target) in enumerate(test_loader):
                    output = flearner(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / (outer_loop_batch_size * len(test_loader))
                print("----------------------------------")
                print("Done training...")
                print("Final test accuracy:", accuracy)

                # Final inner loop training curve
                plt.plot(np.arange(len(inner_losses)), inner_losses)
                plt.xlabel("Inner loop iteration")
                plt.ylabel("Cross entropy loss")
                plt.show()

                break

            loss.backward()

        optimizer_teacher.step()
        break

    torch.save(teacher.state_dict(), 'weights/teacher.pth')


    print("###################################################################################################################")
    print("Running simple neural architecture search")
    best_accuracy = 0
    for i in range(num_architectures):

        # Randomly sample architecture
        conv1_filters = np.random.randint(1, 64)
        conv2_filters = np.random.randint(1, 128)

        learner = Learner(conv1_filters, conv2_filters)
        inner_optim = optim.SGD(learner.parameters(), lr=learner_lr.item(), momentum=learner_momentum.item())
        learner.train()

        # For some reason if we don't use higher here, accuracy drops significantly
        with higher.innerloop_ctx(learner, inner_optim,
                                  override={'lr': [learner_lr], 'momentum': [learner_momentum]}) as (flearner, diffopt):
            for step in range(inner_loop_iterations):

                # Data generation
                if use_curriculum:
                    z_vec = curriculum[step]
                else:
                    z_vec = torch.randn(inner_loop_batch_size, noise_size)

                one_hot = F.one_hot(label, num_classes)

                # Pass input to teacher to generate synthetic images
                teacher_output, teacher_target = teacher(z_vec, one_hot)

                # Pass teacher output to the learner
                learner_output = flearner(teacher_output)
                loss = loss_fn(learner_output, label)
                diffopt.step(loss)

            # Compute accuracy on validation set
            correct = 0
            for val_idx, (data, target) in enumerate(val_loader, 0):
                # if (val_idx == val_iterations): break
                output = flearner(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / (outer_loop_batch_size * len(val_loader))

            if (accuracy > best_accuracy):
                best_accuracy = accuracy
                filter_counts = (conv1_filters, conv2_filters)

            print("------------------------- Architecture", i + 1, " -------------------------")
            print("Num conv1 filters:", conv1_filters, ", Num conv2 filters:", conv2_filters, ", Val accuracy:",
                  accuracy)

            if (i == num_architectures - 1):
                correct = 0
                for test_idx, (data, target) in enumerate(test_loader, 0):
                    # if (test_idx == test_iterations): break
                    output = flearner(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / (outer_loop_batch_size * len(test_loader))
                print("------------------------- Best architecture -------------------------")
                print("Num conv1 filters:", filter_counts[0], ", Num conv2 filters:", filter_counts[1],
                      ", Test accuracy:", accuracy)


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data_loaders(mnist_mean, mnist_std, outer_loop_batch_size)
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)
    test_iterator = iter(test_loader)
    train_and_nas(train_iterator,val_iterator, test_iterator)


