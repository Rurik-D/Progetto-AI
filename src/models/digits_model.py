import cv2
import numpy as np
import os
import pandas as pd
import pickle

from tkinter import filedialog

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchmetrics



def select_file(filepath=None):
    if filepath != None:
        return filepath
    else:
        filepath = filedialog.askopenfilename(title="Select the dataset file")
        return filepath


def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

'''
It deserialize ("unpickle") the dataset object, previously serialized,
from a file. The serialized object is supposed to be encoded as bytes.
'''
def load_dataset1(dataset_path):
    with open(dataset_path, 'rb') as data:
        data = pickle.load(data, encoding="bytes")
    return data

def load_dataset2(df):
    def rinomina_dati(x):
        path_images = "C:\\Users\\alexb\\Downloads\\dg_data"
        x = os.path.join(path_images, x)
        x = cv2.imread(x, 0)
        x = np.array(x, dtype="uint8")
        return x
    df.rename(columns={"filename":"data", "label":"labels"}, inplace=True)
    df['data'] = df['data'].apply(rinomina_dati)
    
    data2 = df.to_dict("list")
    
    data2['labels'] = np.expand_dims(np.array(data2['labels'],dtype="int64"), axis=1)
    data2['data'] = np.expand_dims(np.array(data2['data'], dtype="uint8"), axis=0)
    data2['data'] = data2['data'][0]
    
    return data2


def choose_device(feedback=False):
    device = ("cuda" if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")
    
    if feedback:
        print("Device in use:", device)
    
    return device



class QMNISTDataset(Dataset):
    def __init__(self, data, labels):
        self.images = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        '''
        It convert the image into a float32 format and
        normalize the values in range from 0 to 1.
        '''
        def normalize(image):
            PIXELS_RANGE = 255
            image_float = image.astype(np.float32)
            image_norm = image_float / PIXELS_RANGE
            return image_norm
            
        image = normalize(self.images[idx])
        label = self.labels[idx]
        
        return torch.tensor(image), torch.tensor(label)
    
    
    '''
    Returns the set of possible classes in labels.
    '''
    def get_classes(self):
        return set(int(label.item()) for label in train_dataset.labels)
    
    '''
    Returns the number of possible classes in labels.
    '''
    def get_num_classes(self):
        return len(self.get_classes())


'''
It creates the data loader, defining also the size of each batch to pass
and the shuffling of the data.
'''
def create_dataloader(train, test, batch_size):
    
    train_dataloader = DataLoader(train,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    test_dataloader = DataLoader(test,
                                 batch_size=batch_size,
                                 shuffle=False)
    
    return train_dataloader, test_dataloader


'''
Initialize a model, composed of to main blocks:
a convolutional part and an MLP part.
In the convolutional part there are three convolutional blocks
followed by a batch normalization, an activation function and
a max pooling.
In the MLP part there are three linear layer, respectively
a first layer with 256 hidden units, a second one with 128 hidden
units and a third one with 10 output units (the output classes);
furthermore, both the first and second layer are followed by
an activation function and a 50% dropout (in order to prevent overfitting).
In the transition from the convolutional part to the MLP part,
the multidimensional tensor produced by CNN is flattened, in order to
make it an unidimensional vector.
Hence the convulutional part extracts the features of the image,
while the MLP part takes charge of the final classification.
'''
class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        
        self.mlp = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, train_dataset.get_num_classes())
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

'''
It displays a rich report about the training process.
For each amount of sample (modifiable through "update_freq")
it displays the percentage of completion of an epoch, the current
position in batch, the result of the loss function and an estimate
of the reached accuracy.

'''
def display_train_process(batch, loss, X, y, pred, size, update_freq=4):
    metric.update(pred, y)
    
    if batch % update_freq == 0:
        loss_value, current_batch = loss.item(), (batch + 1) * len(X)
        accuracy = metric(pred, y)
        accuracy_perc = round(float(accuracy)*100, 2)
        
        perc = round(current_batch*100/size, 1)
        loaded = "█"*int(perc//10)
        loading = "█" if perc > 99 else " "*(10-len(loaded))
        
        print(f"|{loaded}{loading}|{perc}%\nCurrent batch: [{current_batch}/{size}]")
        print(f"Loss: {loss_value:>7f}\nAccuracy: {accuracy_perc}%\n")
    

'''
The train loop extracts the feature X and the labels y from the batch
and modifies their dimensions to adapt them to the model.
Then it calculates the loss function and uses it to apply a backpropagation
of gradient. In the end, it updates the weights of gradient through
the optimizer. Before computing the next epoch, it resets the gradients.
'''
def train_loop(dataloader,model,loss_fn,optimizer):
    size = len(train_loader.dataset)
    
    model.train()
    metric.reset()

    for batch, (X,y) in enumerate(dataloader):
    
        X = X.unsqueeze(1)
        y = y.squeeze(1) 

        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
         
        display_train_process(batch, loss, X, y, pred, size)

    accuracy = metric.compute()
    accuracy_perc = round(float(accuracy)*100, 2)
    
    print("_"*40)
    print(f"\tFinal train accuracy: {accuracy_perc}%")
    
    metric.reset()
    

'''
The test loop extracts the feature X and the labels y from the batch
and modifies their dimensions to adapt them to the model.
Then it computes the prediction on model.
'''
def test_loop(dataloader, model):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for X,y in dataloader:
            
            X = X.unsqueeze(1)
            y = y.squeeze(1) 

            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            
            metric.update(pred, y)
    
    accuracy = metric.compute()

    accuracy_perc = round(float(accuracy)*100, 2)
    print(f"\tFinal test accuracy: {accuracy_perc}%")
    print("_"*40, "\n\n")
    
    metric.reset()


'''
It saves the weights of the computed PyTorch model in a specified file.
If "feedback" is set on True, it also displays a confirmation of the saving.
'''
def save_model(feedback=False):
    torch.save(model.state_dict(),"digits_rec(v2).pth") # MOFICARE NOME FILE .PTH
    
    if feedback:
        print("Model saved")



# Loading QMNIST Dataset
data1 = load_dataset1(select_file("C:\\Users\\alexb\\Downloads\\MNIST-120k"))
data2 = load_dataset2(read_csv(select_file("C:\\Users\\alexb\\Downloads\\our_digits.csv")))

data = data1
for key in data2:
    key = np.stack((data[key], data2[key]))

print(len(data['data']))

# Splitting dataset for train and test
train_data, test_data, train_labels, test_labels = train_test_split(data['data'],
                                                                    data['labels'],
                                                                    test_size=0.2,
                                                                    random_state=42)

# Loading customed QMNIST dataset
train_dataset = QMNISTDataset(train_data,train_labels)
test_dataset = QMNISTDataset(test_data,test_labels)

# Defining hyperparameters
EPOCHS = 10
BATCH_SIZE = 2000
LEARNING_RATE = 0.0001

# Defining the device
device = choose_device()

# Creating data loaders
train_loader, test_loader = create_dataloader(train_dataset,
                                              test_dataset,
                                              BATCH_SIZE)

# Defining the model
model = OurCNN().to(device)

# Defining loss function
loss_fn = nn.CrossEntropyLoss()

# Defining optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Defining accuracy metrics
metric = torchmetrics.Accuracy(task='multiclass',
                               num_classes=train_dataset.get_num_classes()).to(device)

# Defining training and test loops
for epoch in range(EPOCHS):
    print("Epoch:", epoch+1)
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model)

# Saving model
save_model()