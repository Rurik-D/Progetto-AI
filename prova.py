
import torch
from torch import nn
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt
# Sposta il tensore sull'hardware appropriato (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 3),
            nn.ReLU(),
            nn.Conv2d(5, 10, 3),
            nn.ReLU()
            )
        self.mlp = nn.Sequential(
            nn.Linear(24*24*10,10),
            nn.ReLU(),
            nn.Linear(10,10)
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
transform = Resize((28, 28))  # Ridimensiona l'immagine alle dimensioni di input del modello
transform = ToTensor()     # Converte l'immagine in un tensore

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