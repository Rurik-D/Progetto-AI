import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchmetrics
import torch.nn.functional as F
# Leggi il file CSV
csv_input = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\our_dataset.csv'
df = pd.read_csv(csv_input)


train_df, test_df = train_test_split(df, test_size=0.2, random_state=100)

path_images = "C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug"

train_df['filename'] = train_df['filename'].apply(lambda x: os.path.join(path_images, x))
test_df['filename'] = test_df['filename'].apply(lambda x: os.path.join(path_images, x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = int(self.dataframe.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Trasformazioni delle immagini
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Crea i DataLoader per il training e il test set
train_dataset = CustomImageDataset(train_df, transform=transform)
test_dataset = CustomImageDataset(test_df, transform=transform)

epochs = 5
batch = 30
learning_rate = 0.001

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

# Definisci il modello CNN
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
        x = self.fc2(x)  # Do not apply sigmoid here
        return x

model = OurCNN().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

metric = torchmetrics.Accuracy(task='binary').to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
    for epoch in range(epochs):
        model.train()
        metric.reset()  # 

        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()

            pred = model(images)
            loss = loss_fn(pred, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            metric.update(torch.sigmoid(pred), labels)
        
            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch * len(images), len(dataloader.dataset),
                    100. * batch / len(dataloader), loss.item()))
    
        acc = metric.compute()
        print(f'Final Accuracy: {acc}')
        metric.reset()

def test_loop(dataloader, model):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()
            pred = model(images)
            metric.update(torch.sigmoid(pred), labels)
    
    acc = metric.compute()

    print(f'Final Testing accuracy: {acc}')
    metric.reset()

for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model)

#torch.save(model.state_dict(),"sudoku_rec_model.pth")

print("Done!")