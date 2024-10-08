import cv2
import numpy as np
from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchmetrics



def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df


def load_dataset(df):
    def rinomina_dati(x):
        path_images = path.abspath(".") + f"\\dataset\\Printed digits\\dg_data"
        x = path.join(path_images, x)
        x = cv2.imread(x, 0)
        x = np.array(x, dtype="uint8")
        return x
    df.rename(columns={"filename":"data", "label":"labels"}, inplace=True)
    df['data'] = df['data'].apply(rinomina_dati)
    
    data = df.to_dict("list")
    
    data['labels'] = np.expand_dims(np.array(data['labels'],dtype="int64"), axis=1)
    data['data'] = np.expand_dims(np.array(data['data'], dtype="uint8"), axis=0)
    data['data'] = data['data'][0]
    
    return data


def choose_device(feedback=False):
    device = ("cuda" if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")
    
    if feedback:
        print("Device in use:", device)
    
    return device



class OurDataset(Dataset):
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
    Returns the number of possible classes in labels.
    '''
    def get_num_classes(self):
        NUM_CLASSES = 10
        return NUM_CLASSES


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
            
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x
    
# Defining the device
device = choose_device()
# Defining the model
model = OurCNN().to(device)


'''
It displays a rich report about the training process.
For each amount of sample (modifiable through "update_freq")
it displays the percentage of completion of an epoch, the current
position in batch, the result of the loss function and an estimate
of the reached accuracy.

'''
def display_train_process(batch, loss, X, y, pred, size, metric, update_freq=4):
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
def train_loop(dataloader,model,loss_fn,optimizer, metric,train_loader):
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
         
        display_train_process(batch, loss, X, y, pred, size, metric)

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
def test_loop(dataloader, model, metric):
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
    torch.save(model.state_dict(),"digits_model.pth") # MOFICARE NOME FILE .PTH
    
    if feedback:
        print("Model saved")




# Loading QMNIST Dataset
def test_labels_data():
    data = load_dataset(read_csv(path.abspath(".") + f"\\dataset\\Printed digits\\our_digits.csv"))

    # Splitting dataset for train and test
    train_data, test_data, train_labels, test_labels = train_test_split(data['data'],
                                                                        data['labels'],
                                                                        test_size=0.2,
                                                                        random_state=42)
    return train_data, test_data, train_labels, test_labels
# Loading customed QMNIST dataset

def generate_model(train_data, test_data, train_labels, test_labels):
    train_dataset = OurDataset(train_data,train_labels)
    test_dataset = OurDataset(test_data,test_labels)

    # Defining hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 200
    LEARNING_RATE = 0.0001

    # Creating data loaders
    train_loader, test_loader = create_dataloader(train_dataset,
                                                test_dataset,
                                                BATCH_SIZE)

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
        train_loop(train_loader, model, loss_fn, optimizer, metric, train_loader)
        test_loop(test_loader, model, metric)

    # Saving model
    save_model()