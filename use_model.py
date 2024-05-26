import torch
from digits_rec_model import OurCNN  # Assicurati che il file model.py sia nello stesso directory o nel PYTHONPATH
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Verifica se CUDA è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformazioni per i dati di test
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Carica i dati di test
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Istanzia il modello e carica i pesi salvati
model = OurCNN().to(device)
model.load_state_dict(torch.load('digits_rec.pth'))
model.eval()  # Metti il modello in modalità di valutazione

# Verifica del modello con un esempio di previsione
example_data, example_target = next(iter(test_loader))
example_data, example_target = example_data.to(device), example_target.to(device)

# Fai una previsione
with torch.no_grad():
    output = model(example_data)
pred = output.argmax(dim=1, keepdim=True)

# Stampa i risultati delle previsioni
print(f'Predicted: {pred.view(-1).cpu().numpy()}')
print(f'Actual: {example_target.view(-1).cpu().numpy()}')

# Calcola l'accuratezza sul batch di test
correct = pred.eq(example_target.view_as(pred)).sum().item()
accuracy = 100. * correct / len(example_target)
print(f'Accuracy: {accuracy:.2f}%')