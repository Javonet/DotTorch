# visualize_data.py
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class VisualizeData:
    _type_ = "VisualizeData"

    def visualize(self):
        # Function to display images
        def imshow(img):
            img = img / 2 + 0.5  # Unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        # Setup data loader
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

        # Display a batch of images
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        imshow(torchvision.utils.make_grid(images))
