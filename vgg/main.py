import argparse
from pprint import pprint
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import *
from datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def config():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", default="train", type=str, help="mode")

    p.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    p.add_argument("--batch_size", default=32, type=int, help="batch size")
    p.add_argument("--num_epochs", default=50, type=int, help="number of epochs")

    p.add_argument("--data_dir", default="./datasets", help="directory of dataset")

    p.add_argument("--architecture", default="VGG16", help="architecture")

    return p.parse_args()

def main(config):
    data_dir = config.data_dir

    if config.mode == "train":
        pprint(vars(config))
        train_dataset = Dataset(root=data_dir+"/train")
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        val_dataset = Dataset(root=data_dir+"/val")
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

        train_size = len(train_dataset)
        val_size = len(val_dataset)

        model = VGG(in_channels=3, num_classes=len(train_dataset.classes), architecture=config.architecture).to(device)
        
        if len(train_dataset.classes) == 2:
            fn_loss = nn.BCELoss()
        else:
            fn_loss = nn.CrossEntropyLoss()

        optim = torch.optim.Adam(model.parameters(), lr=config.lr)

        start = 0
        for epoch in range(start+1, config.num_epochs+1):
            print("Epoch {}/{}".format(epoch, config.num_epochs))
            print("-"*10)
            model.train()

            running_loss = 0.0
            running_correct = 0

            for inputs, outputs in tqdm(train_loader):

                inputs = inputs.to(device)
                labels = outputs.to(device)
                
                if len(train_dataset.classes) == 2:
                    outputs = model(inputs).squeeze()

                    optim.zero_grad()

                    loss = fn_loss(outputs, labels.to(torch.float32))
                    loss.backward()

                    optim.step()

                    outputs[outputs>=0.5] = 1
                    outputs[outputs<0.5] = 0
                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    optim.zero_grad()
                    
                    loss = fn_loss(outputs, labels)
                    loss.backward()
                    
                    outputs = preds

                    optim.step()


                running_loss += loss.item() * config.batch_size
                running_correct += torch.sum(outputs == labels.data).to(int)
                
            
            epoch_loss = running_loss / train_size
            epoch_acc = running_correct.double() / train_size

            print("Train Loss: {:.4f}\tAccuracy: {:.4f}".format(epoch_loss, epoch_acc))

            running_loss = 0.0
            running_correct = 0

            with torch.no_grad():
                model.eval()

                for inputs, outputs in tqdm(val_loader):
                    inputs = inputs.to(device)
                    labels = outputs.to(device)

                    if len(train_dataset.classes) == 2:
                        outputs = model(inputs).squeeze()
                        loss = fn_loss(outputs, labels.to(torch.float32))

                        outputs[outputs>=0.5] = 1
                        outputs[outputs<0.5] = 0
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = fn_loss(outputs, labels)

                        outputs = preds
                    
                    running_loss += loss.item() * config.batch_size
                    running_correct += torch.sum(outputs == labels.data)
                    
                epoch_loss = running_loss / val_size
                epoch_acc = running_correct.double() / val_size

                print("Valid Loss: {:.4f}\tAccuracy: {:.4f}".format(epoch_loss, epoch_acc))

                torch.save(model, "./model.pth")

    elif config.mode == "test":
        test_dataset = Dataset(root=data_dir+"/test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        test_size = len(test_dataset)

        model = torch.load("./model.pth").to(device)

        if len(test_dataset.classes) == 2:
            fn_loss = nn.BCELoss()
        else:
            fn_loss = nn.CrossEntropyLoss()

        running_loss = 0.0
        running_correct = 0

        with torch.no_grad():
            model.eval()

            for inputs, outputs in tqdm(test_loader):
                inputs = inputs.to(device)
                labels = outputs.to(device)

                if len(test_dataset.classes) == 2:
                    outputs = model(inputs).squeeze()
                    loss = fn_loss(outputs, labels.to(torch.float32))
                    outputs[outputs>=0.5] = 1
                    outputs[outputs<0.5] = 0
                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = fn_loss(outputs, labels)
                    outputs = preds
                
                running_loss += loss.item() * config.batch_size
                running_correct += torch.sum(outputs == labels.data)
                
            epoch_loss = running_loss / test_size
            epoch_acc = running_correct.double() / test_size
            print("Test Loss: {:.4f}\tAccuracy: {:.4f}".format(epoch_loss, epoch_acc))

    else:
        pass

if __name__ == "__main__":
    config = config()
    main(config)
        
