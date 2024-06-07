import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
def choose_device(feedback=False):
    device = ("cuda" if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")
    
    if feedback:
        print("Device in use:", device)
    
    return device

device = choose_device()

class OurCNN(nn.Module):
    def __init__(self):
        super(OurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

model = OurCNN().to(device)
model.load_state_dict(torch.load('sudoku_rec_model.pth'))
model.eval()

image_path = "C:\\Users\\giuse\\Desktop\\Progetto-AI\\img-random.jpeg"

image = cv2.imread(image_path)
image = cv2.boxFilter(image, -1, (8, 8))
#image = cv2.bilateralFilter(image, 9, 75, 75) # in forse
image = cv2.resize(image, (128,128), interpolation=cv2.INTER_AREA)

transform = transforms.Compose([transforms.ToTensor()])

image_tensor = transform(image).unsqueeze(0)

image_pil = transforms.ToPILImage()(image_tensor.squeeze(0))

image_np = image_tensor.squeeze(0).numpy()

plt.imshow(image_np[0], cmap='gray')
plt.show()
image_tensor = image_tensor.to(device)

# Passa il tensore attraverso il modello per ottenere le previsioni delle classi
model.eval()  # Imposta il modello in modalità di valutazione
with torch.no_grad():
    outputs = model(image_tensor)

print(outputs)

_, predicted = torch.max(outputs, 1)
print(predicted)

print("Classe predetta:", predicted.item())

# Mostra l'immagine
plt.imshow(image)
plt.axis('off')
plt.show()