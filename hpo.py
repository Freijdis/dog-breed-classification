import os
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, args):
    logger.info("Begin testing")
    test_loader = _get_test_data_loader(args)

    loss_criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)
        )
    )

    print(f"ACCURACY: {accuracy}")
    print(f"TEST LOSS: {test_loss}")


def freeze_layers_except_last(model):
    logger.info("Freezing all model layers except last one")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def train(args):
    logger.info("Begin model training")

    loss_criterion = nn.CrossEntropyLoss()
    train_loader = _get_train_data_loader(args)

    model = net(args)
    model = freeze_layers_except_last(model)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    return model


def net(args):
    logger.info("Getting the ResNet50 model")
    model = models.resnet50(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, args.num_dog_classes)

    return model


def _get_train_data_loader(args):
    logger.info("Get train data loader")
    train_local_path = args.train_dir

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_local_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader


def _get_test_data_loader(args):
    logger.info("Get test data loader")
    test_local_path = args.test_dir

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_local_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return test_loader


def main(args):

    model = train(args)
    test(model, args)

    logger.info("Saving the model")
    torch.save(model, args.model_path)
    logger.info(f"Model saved under {args.model_path}")


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001, help="Lernrate des Modells")
    parser.add_argument('--batch_size', type=int, default=64, help="Größe der Trainings-Batches")
    parser.add_argument('--epochs', type=int, default=10, help="Anzahl der Epochen")

    parser.add_argument('--num_dog_classes', type=int, default=133, help="Anzahl der Hunderassen")
    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./dogImages/train"), help="Pfad zu den Trainingsdaten")
    parser.add_argument("--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "./dogImages/test"), help="Pfad zu den Testdaten")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_DATA", "./dogImages"), help="Pfad zu den Daten")
    parser.add_argument('--model_path', type=str, default='./models', help="Verzeichnis zum Speichern des Modells")
    parser.add_argument('--test_batch_size', type=int, default=32, help="Größe der Test-Batches")

    args = parser.parse_args()

    main(args)
