import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
# You might not have tqdm, which gives you nice progress bars
# !pip install tqdm
import tqdm
import os
import copy
import pandas as pd
import PIL

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")
    print("You may want to try to use the GPU in Google Colab by clicking in:")
    print("Runtime > Change Runtime type > Hardware accelerator > GPU.")
print(torch.__version__)


random_p = 0.25

class Classifier_Training():
    def __init__(self, number_epochs = 50 , lr = 0.0001, batch_size = 32, save_dir = '/home/ryznerf/models/') -> None:
        self.model_name = "resnet50"
        self.num_classes = 2
        self.batch_size = batch_size
        self.num_epochs = number_epochs
        self.save_dir = save_dir
        self.learning_rate = lr
        os.makedirs(self.save_dir, exist_ok=True)

        self.resume_from = None
        self.pretrained = True
        self.save_all_epochs = False
        self.early_stopping = True
        self.shuffle_datasets = True

    def train_model(self, dataset_dir = '/home/ryznerf/Computer_Vision/pytorch_data_input', model_name = 'test_model'):
        from functools import partial
        from tqdm import tqdm
        tqdm = partial(tqdm, position=0, leave=True)
        dataset_dir = dataset_dir
        model_1, input_size = initialize_model(model_name=self.model_name, num_classes=self.num_classes, resume_from=self.resume_from, use_pretrained=self.pretrained)
        dataloaders = get_dataloaders(dataset_dir, input_size, self.batch_size, self.shuffle_datasets)
        criterion = get_loss()
        model_1 = model_1.to(device)
        optimizer_1 = make_optimizer(model_1, self.learning_rate, 1)

        trained_model_1, validation_history_1, train_history_1 = train_model(model=model_1,
                                                                             dataloaders=dataloaders,
                                                                             criterion=criterion,
                                                                             optimizer=optimizer_1,
                                                                             save_dir=self.save_dir,
                                                                             save_all_epochs= self.save_all_epochs,
                                                                             num_epochs= self.num_epochs)
        torch.save(trained_model_1, self.save_dir + model_name)

        return trained_model_1, validation_history_1, train_history_1

### FUNCTIONS ---------------------------------------------------------------------------------------------------------
def initialize_model(model_name, num_classes, resume_from=None, use_pretrained=False):
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        input_size = 224

    return model_ft, input_size


from torchvision.transforms.functional import to_grayscale

def get_training_image_transforms():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=(-30, 30), translate=(0, 0.2)),
        transforms.RandomHorizontalFlip(p=random_p),
        transforms.RandomVerticalFlip(p=random_p),
        #            transforms.RandomApply(transforms = [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p = 0.1),
        #            transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.5)], p=random_p),
        transforms.ToTensor(),
        transforms.Normalize([0.0949], [0.1951]),
        #            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform


def get_image_transforms():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.0857], [0.1818]),
        #            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def get_dataloaders(dataset_dir, input_size, batch_size, shuffle=True, transform=get_image_transforms(),
                    training_transform=get_training_image_transforms()):
    data_transforms = {
        'train': training_transform,
        'val': transform,
        'test': transform
    }
    # Create training, validation and test datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x]) for x in
                      data_transforms.keys()}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                       shuffle=False if x != 'train' else shuffle, num_workers=4) for x
                        in data_transforms.keys()}
    return dataloaders_dict


def train_model(model, dataloaders, criterion, optimizer, save_dir=None, save_all_epochs=False, num_epochs=25):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # TQDM has nice progress bars
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # torch.max outputs the maximum value, and its index
                    # Since the input is batched, we take the max along axis 1
                    # (the meaningful outputs)
                    _, preds = torch.max(outputs, 1)

                    # backprop + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_acc_history.append(epoch_acc)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if save_all_epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, f'weights_{epoch}.pt'))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save and load best model weights
    torch.save(best_model_wts, os.path.join(save_dir, 'weights_best_val_acc.pt'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'weights_last.pt'.format(epoch)))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history


def make_optimizer(model, learning_rate, weight_decay=0, print_parameters=False):
    # Get all the parameters
    params_to_update = model.parameters()
    if print_parameters:
        print("Params to learn:")
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    #    optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay = weight_decay)
    optimizer = torch.optim.AdamW(params_to_update, lr=learning_rate)
    return optimizer


def get_loss():
    # Create an instance of the loss function
    criterion = nn.CrossEntropyLoss()
    return criterion


##### MAIN
if __name__ == "__main__":
    # determine the normalization parameters
    model_wrapper = Classifier_Training()
    model, _, _ = model_wrapper.train_model()