# train.py
import os
import string
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

class Train:
    __type__ = "Train"
    
    def train(self, models_path: string):
        # Setup data loaders
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

        # Initialize model, loss, and optimizer
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Training loop
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        # Save the trained model
        os.makedirs(models_path, exist_ok=True)
        torch.save(model.state_dict(), models_path + '/cifar10_model.pth')
        print(f"Model saved in '{models_path}/cifar10_model.pth'")
