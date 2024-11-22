# test_model.py
import string
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

class TestModel:
    __type__ = "TestModel"
    
    def test(self, models_path: string):
        # Setup data loader for testing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

        # Load the trained model
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(models_path + '/cifar10_model.pth'))
        model.eval()

        # Test the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
