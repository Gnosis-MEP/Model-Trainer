import os
from matplotlib import pyplot


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms.transforms import InterpolationMode

from model_trainer.cls_model_transforms import get_transforms
from model_trainer.dataloader_torch import get_train_test_split
from model_trainer.conf import DATASET_PATH, MODELS_PATH, MODEL_ID
from torch.utils.data import (
    DataLoader,
)


def get_base_fine_tuned_model(base_model, num_classes=2, freeze=True):
    # # Initialize model
    # weights = models.MobileNet_V3_Large_Weights.DEFAULT
    # model = models.get_model("mobilenet_v3_large", weights=weights)


    if freeze:
        # freeze all layers, change final linear layer with num_classes
        for param in base_model.parameters():
            param.requires_grad = False

    num_classes = 2
    base_model.classifier[-1] = nn.Linear(base_model.classifier[-1].in_features, num_classes)
    return base_model



def fine_tune(num_epochs=1):

    device = torch.device('cuda')

    train_transform = get_transforms('TRAIN')
    val_transform = get_transforms('VAL')

    train_dataset, val_dataset = get_train_test_split(DATASET_PATH, train_transform=train_transform, val_transform=val_transform)

    # # Initialize model
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.get_model("mobilenet_v3_large", weights=weights)

    # # # Initialize inference transforms
    # # preprocess = weights.transforms()

    # # freeze all layers, change final linear layer with num_classes
    # for param in model.parameters():
    #     param.requires_grad = False

    # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    num_classes = 2
    model = get_base_fine_tuned_model(model, num_classes)
    model.to(device)

    # Hyperparameters
    learning_rate = 3e-4
    batch_size = 32
    # num_epochs = 1  # You can adjust the number of epochs
    weight_decay = 1e-5

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train Network
    for epoch in range(num_epochs):
        train_losses = []

        for batch_idx, (data, targets) in enumerate(train_data_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            train_losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            print(f"Current batch: {batch_idx}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_losses = []
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in val_data_loader:
                data, targets = data.to(device=device), targets.to(device=device)

                # Forward pass and compute validation loss
                val_scores = model(data)
                val_loss = criterion(val_scores, targets)
                val_losses.append(val_loss.item())

                # Calculate accuracy
                _, predicted = torch.max(val_scores, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        val_accuracy = (correct_predictions / total_samples) * 100
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.2f}%")

        # Set the model back to training mode
        model.train()

    # print("Checking accuracy on Training Set")
    # check_accuracy(device, train_data_loader, model)
    # print("Checking accuracy on Val Set")
    # check_accuracy(device, val_data_loader, model)

    model_path = os.path.join(MODELS_PATH, f'{MODEL_ID}.pth')
    torch.save(model.state_dict(), model_path)
    return model


# Check accuracy on training to see how good our model is
def check_accuracy(device, loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


if __name__ == '__main__':
    fine_tune(1)
