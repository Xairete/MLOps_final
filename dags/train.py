import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import boto3
from torchvision import transforms
from PIL import Image
import os
import glob
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

def _train(model, optimizer, sheduler, loss_fn, train_loader, val_loader, epochs=5, device="cpu"):
    max_acc = 0
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        n_train = len(train_loader)
        for batch in tqdm(train_loader, total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img'):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        sheduler.step()

        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in tqdm(val_loader, total=len(val_loader), desc=f' Validation Epoch {epoch + 1}/{epochs}', unit='img'):
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
                        
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)
        if (num_correct / num_examples) > max_acc:
            max_acc = num_correct / num_examples
            torch.save(model.state_dict(), "model.pth")
            print('Model saved!')
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
        
def upload_model_and_remove_dir():
    if os.path.isfile("model.pth"):
        session = boto3.session.Session(aws_access_key_id = '***', 
                                        aws_secret_access_key='***')

        s3 = session.client(
                service_name='s3',
                endpoint_url='https://storage.yandexcloud.net')
        s3.upload_file("train_list.csv",'mlops-dev-german',  "model.pth")
        print('Model uploaded!')
        os.remove("model.pth")
    if os.path.exists("./data"):
        shutil.rmtree("./data")
        print('Train data removed!')
        
class dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path.split('/')[-1].split('_')[0]
            
        return img_transformed, int(label)

def train():
    print('Train started')
    train_dir = "./data"
    train_list = glob.glob(os.path.join(train_dir,'*.png'))
    if len(train_list) > 0 and not os.path.isfile("model.pth"): 
        
        model_resnet18 = models.resnet50(pretrained=True)
        for name, param in model_resnet18.named_parameters():
            if("bn" not in name):
                param.requires_grad = False

        num_classes = 2
        model_resnet18.fc = nn.Sequential(nn.Linear(model_resnet18.fc.in_features,512),
                                          nn.ReLU(),
                                          nn.Dropout(),
                                          nn.Linear(512, num_classes))
        img_dimensions = 224

        img_transforms = transforms.Compose([
            transforms.Resize(img_dimensions),
            transforms.CenterCrop(img_dimensions),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
            ])
        if torch.cuda.is_available():
            device = torch.device("cuda") 
        else:
            device = torch.device("cpu")
        model_resnet18.to(device)
        random.shuffle(train_list)
        train_list, val_list = train_test_split(train_list, test_size=0.2)
        train_data = dataset(train_list[:300], transform=img_transforms)
        val_data = dataset(val_list, transform=img_transforms)
        batch_size = 4

        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        validation_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(model_resnet18.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        _train(model_resnet18, optimizer, scheduler, torch.nn.CrossEntropyLoss(), train_data_loader, validation_data_loader, epochs=3, device=device)
        
    upload_model_and_remove_dir()