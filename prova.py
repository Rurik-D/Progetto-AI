
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# Sposta il tensore sull'hardware appropriato (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # nn.Conv2d(1, 5, 3),
            # nn.ReLU(),
            # nn.Conv2d(5, 10, 3),
            # nn.ReLU(),
            # nn.Conv2d(10, 1, 3),
            # nn.ReLU()
            nn.Conv2d(1, 32, 3, padding=1),  # Primo strato convoluzionale
            nn.BatchNorm2d(32),               # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Primo strato di pooling
            nn.Conv2d(32, 64, 3, padding=1),  # Secondo strato convoluzionale
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Secondo strato di pooling
            nn.Conv2d(64, 128, 3, padding=1), # Terzo strato convoluzionale
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                # Terzo strato di pooling
            )
        self.mlp = nn.Sequential(
            # nn.Linear(22*22,10),
            # nn.ReLU(),
            # nn.Linear(10,10)
            nn.Linear(128 * 3 * 3, 256),      # Primo strato lineare
            nn.ReLU(),
            nn.Dropout(0.5),                  # Dropout
            nn.Linear(256, 128),              # Secondo strato lineare
            nn.ReLU(),
            nn.Dropout(0.5),                  # Dropout
            nn.Linear(128, 10)    
        )

    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape)
        x = torch.flatten(x,1)
        #print(x.shape)
        x = self.mlp(x)
        return x
    
model = OurCNN().to(device)
model.load_state_dict(torch.load('digits_rec.pth'))
model.eval()

image_path = "C:\\Users\giuse\Desktop\Progetto-AI\prova.jpg"
image = Image.open(image_path)


# Applica le trasformazioni all'immagine
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Ridimensiona l'immagine alle dimensioni di input del modello
    transforms.ToTensor()        # Converte l'immagine in un tensore
])

# Applica le trasformazioni e aggiunge una dimensione di batch
image_tensor = transform(image).unsqueeze(0)

image_tensor = image_tensor.to(device)

# Passa il tensore attraverso il modello per ottenere le previsioni delle classi
model.eval()  # Imposta il modello in modalità di valutazione
with torch.no_grad():
    outputs = model(image_tensor)

print(outputs)
# Ottieni le previsioni delle classi
_, predicted = torch.max(outputs, 1)
print(predicted)
# Stampa le previsioni
print("Classe predetta:", predicted.item())

# Mostra l'immagine
plt.imshow(image)
plt.axis('off')
plt.show()