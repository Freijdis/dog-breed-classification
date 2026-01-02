import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import ImageFile
import smdebug.pytorch as smd

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, hook):
    logger.info("Begin testing")

    loss_criterion = nn.CrossEntropyLoss()
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            test_loss += loss_criterion(output, target).item()
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)
        )
    )


def freeze_layers_except_last(model):
    logger.info("Freezing all model layers except last one")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


"""def train(model, train_loader, valid_loader, optimizer, epoch, hook):
    logger.info("Begin model training")

    loss_criterion = nn.CrossEntropyLoss()
    hook.set_mode(smd.modes.TRAIN)

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 2:
            break
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )"""


def train(model, train_loader, valid_loader, optimizer, epochs, hook):
    logger.info("Begin model training")

    loss_criterion = nn.CrossEntropyLoss()
    image_dataset = {'train': train_loader, 'valid': valid_loader}
    hook.set_mode(smd.modes.TRAIN)

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch + 1}/{epochs}")
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(image_dataset[phase]):
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_dataset[phase].dataset)
            epoch_acc = running_corrects.double() / len(image_dataset[phase].dataset)

            logger.info('{} Train Loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    return model


def net(args):
    logger.info("Getting the ResNet50 model")
    model = models.resnet50(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, args.num_dog_classes)

    model = freeze_layers_except_last(model)

    return model


def _get_data_loaders(args):
    logger.info("Get train data loader")
    train_local_path = args.train_dir
    valid_local_path = args.valid_dir
    test_local_path = args.test_dir

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_local_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = datasets.ImageFolder(root=valid_local_path, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)

    test_dataset = datasets.ImageFolder(root=test_local_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def main(args):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Running on Device {device}")

    model = net(args)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    train_loader, valid_loader, test_loader = _get_data_loaders(args)

    optimizer = optim.Adam(model.parameters())

    # for epoch in range(1, args.epochs + 1):
    model = train(model, train_loader, valid_loader, optimizer, args.epochs, hook)  # , device)
    test(model, test_loader, hook)

    logger.info("Saving the model")
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_path, "model.pth"))
    logger.info(f"Model saved under {args.model_path}")


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001, help="Lernrate des Modells")
    parser.add_argument('--batch_size', type=int, default=64, help="Größe der Trainings-Batches")
    parser.add_argument('--epochs', type=int, default=10, help="Anzahl der Epochen")

    parser.add_argument('--num_dog_classes', type=int, default=133, help="Anzahl der Hunderassen")
    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./dogImages/train"), help="Pfad zu den Trainingsdaten")
    parser.add_argument("--valid_dir", type=str, default=os.environ.get("SM_CHANNEL_VALID", "./dogImages/valid"), help="Pfad zu den Testdaten")
    parser.add_argument("--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "./dogImages/test"), help="Pfad zu den Testdaten")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_DATA", "./dogImages"), help="Pfad zu den Daten")
    parser.add_argument('--model_path', type=str,  default=os.environ.get("SM_MODEL_DIR", '/opt/ml/model'), help="Verzeichnis zum Speichern des Modells")
    parser.add_argument('--valid_batch_size', type=int, default=32, help="Größe der Validation-Batches")
    parser.add_argument('--test_batch_size', type=int, default=32, help="Größe der Test-Batches")

    args = parser.parse_args()

    main(args)