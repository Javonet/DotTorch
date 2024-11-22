# test_custom_image.py
import string
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

class TestCustomImage:
    __type__ = "TestCustomImage"
    
    def test(self, models_path: string, image_path: string) -> string:
        # Load the trained model
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
        model.load_state_dict(torch.load(models_path + '/cifar10_model.pth'))
        model.eval()

        # Define the classes for CIFAR-10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Define the image transformation to match training setup
        transform = transforms.Compose([
            transforms.Resize((32, 32)),        # Resize image to 32x32
            transforms.ToTensor(),              # Convert image to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
        ])

        # Load and preprocess your custom image
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Run the image through the model
        output = model(image)
        _, predicted = torch.max(output, 1)

        result = classes[predicted.item()]

        # Print the prediction
        print(f"Predicted class by Python: {result}")
        
        return result
