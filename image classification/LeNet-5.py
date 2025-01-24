import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision import transforms

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(576, 120),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):   
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        
        logits = self.fc3(out)
  
        return logits
    
def main():
    print("Is CUDA available:", torch.cuda.is_available())
    
    # build dataset and dataloader for train/test
    train_dataset = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = datasets.CIFAR10('cifar', False, transform=transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # get a train sample: data + label
    x, label = next(iter(train_dataloader))
    print('x:', x.shape, 'label:', label.shape)
    
    # build model/loss_func/optimizer on available device (GPU or CPU) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    
    # if checkpoint exists
    start_epoch = 0
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
            
    # train & eval process
    for epoch in range(start_epoch+1, 50):
        
        # train
        model.train()
        for x, label in train_dataloader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # eval 
        model.eval()
        with torch.no_grad():
            correct_sample = 0
            total_sample = 0
            for x, label in test_dataloader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct_sample += torch.eq(pred, label).float().sum().item()
                total_sample += x.size(0)
            acc = correct_sample / total_sample
        
        print('epoch:{:2d}, train_loss:{:.4f}, eval_accuracy:{:.4f}'.format(epoch, loss, acc))
        
        # save checkpoint
        if epoch > 0 and epoch % 10 == 0:
            print('save checkpoint')
            torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, "checkpoint.pth")
        
if __name__ == '__main__':
    main()