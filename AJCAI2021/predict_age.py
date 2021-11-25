import os
import torchvision
from matplotlib import pyplot as plt
import torch
import glob
import os
from shutil import copyfile
from io import BytesIO
from PIL import Image
import time
import heapq
from heapq import heappush
from heapq import heappop
from torchvision import datasets
import sys


import gc
gc.collect()
torch.cuda.empty_cache() 


checkpoint_path = './models/model_best_age.pth.tar'
# checkpoint_path = 'checkpoint1283.pth.tar'
data_folder = "D:\\MaskedFaceRecognitionCompetition\\dataset\\UTKface_inthewild-20210331T075050Z-001\\UTKface_inthewild\\classification\\age_separate\\test"

num_classes = 1
shape = 224
BATCHSIZE = 1


transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(shape),
        torchvision.transforms.CenterCrop(shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arch = 'resnet50'
model = torchvision.models.__dict__[arch](pretrained=False)
model.fc = torch.nn.Linear(2048, num_classes)
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()

state_dict = torch.load(checkpoint_path)

parallel_model = torch.nn.DataParallel(model)
parallel_model.load_state_dict(
    torch.load(checkpoint_path, map_location=str(device))['state_dict']
)

model = parallel_model.module

#print(model)

#model.nograd()
model.eval()
model.to(device)

data_transforms = {
    'predict': transform 
    }


dataset = datasets.ImageFolder(data_folder, transform=transform)
dataloader = {'predict': torch.utils.data.DataLoader(dataset, batch_size = BATCHSIZE , shuffle=False, pin_memory=True)}

count = 0
total_correct = 0
total_correctsq = 0
with torch.no_grad():
  for inputs, labels in dataloader['predict']:
    count += 1
    inputs = inputs.to(device)
    output = model(inputs)
    output = output.to(device)
    out_labels = output*60 + 60
    out_labels = torch.transpose(out_labels, 0, 1).to(device)
    labels = labels.to(device)
    #print(labels)
    #print(out_labels)
    correct = (abs(out_labels - labels)).float().sum().to(device)
    correct_sq = (abs(out_labels - labels)**2).float().sum()
    #print(correct)
    #print(out_labels)
    total_correct += correct
    total_correctsq += correct_sq
    #print(out_labels)

print("MAE = " + str(total_correct/len(dataloader['predict'].dataset)))
print("RMSE = " + str((total_correctsq/len(dataloader['predict'].dataset))**0.5))
#print("Accuracy = " + str(total_correct) + "/" + str(len(dataloader['predict'].dataset)))
#print(" ===  " + str(total_correct/len(dataloader['predict'].dataset)) + "  ===")
