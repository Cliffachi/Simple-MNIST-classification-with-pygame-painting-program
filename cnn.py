import torch
import torchvision
import numpy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from PIL import Image
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#loader dataset
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307],
                         std=[0.0307])
])

dataset = './data/'

train_set = torchvision.datasets.MNIST(root=dataset, train=True, download=True, transform=transform)
test_set  = torchvision.datasets.MNIST(root=dataset, train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size = 32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size = 32, shuffle=False)

# mlp and cnn class
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # gray scale image = 1 in_channel, out_channels = 32 features
        # kernel_size = 3 and padding = (kernel_size - 1)/2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 1 x 28 x 28 -> 32 x 14 x 14

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64 x 7 x 7

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.maxpool2(self.relu(self.conv2(x)))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = CNN()
cnn_model.to(device=device)
# train_model(cnn_model, train_data=train_loader, num_epochs=10, lr=0.001)
def predict_image(model, image, device=device):
    model.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        prediction = logits.argmax(dim=1)

    return prediction.item()