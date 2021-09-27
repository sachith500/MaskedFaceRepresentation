import gc

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets

gc.collect()
torch.cuda.empty_cache()


class MyImageFolder(datasets.ImageFolder):
    def getitem(self, index):
        return super(MyImageFolder, self).getitem(index), self.imgs[index]  # return image path


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


checkpoint_path = 'model_best.pth.tar'
data_folder = "D:\\MaskedFaceRecognitionCompetition\\dataset\\UTKface_inthewild-20210331T075050Z-001\\UTKface_inthewild\\classification\\gender_separate\\test"

num_classes = 2
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

# state_dict = torch.load(checkpoint_path)

# parallel_model = torch.nn.DataParallel(model)
checkpoints = torch.load(checkpoint_path, map_location=str(device))['state_dict']
new_checkpoints = {}
for key in checkpoints.keys():
    new_key = key.replace('module.', '')
    new_checkpoints[new_key] = checkpoints[key]

print(model.eval())
model.load_state_dict(new_checkpoints)

print(model)

model.eval()
model.to(device)

data_transforms = {
    'predict': transform
}

# MyImageFolder
# dataset = datasets.ImageFolder(data_folder, transform=transform)#
dataset = ImageFolderWithPaths(data_folder, transform=transform)
dataloader = {'predict': torch.utils.data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=True)}

count = 0
total_correct = 0
with open('submission.txt', 'w') as the_file:
    the_file.write('Name, Label, Predicted' + "\n")
    for inputs, labels, filenames in dataloader['predict']:
        # print(filenames)
        count += 1
        if (count % (1000) == 0):
            print(count)
            # print(time.time() - since)
        inputs = inputs.to(device)
        output = model(inputs)
        output = output.to(device)
        out_labels = torch.argmax(output, dim=1)
        out_labels = out_labels.to('cpu')
        labels = labels.to('cpu')
        correct = (out_labels == labels).float().sum()
        for i in range(len(output)):
            # out_str = str(count) + "_" + str(i) + "," + str(labels[i].item()) + "," + str(out_labels[i].item())
            out_str = filenames[i].split("/")[-1] + "," + str(labels[i].item()) + "," + str(out_labels[i].item())
            the_file.write(out_str + "\n")

        # print(correct)
        total_correct += correct
        # print(out_labels)

print("Accuracy = " + str(total_correct) + "/" + str(len(dataloader['predict'].dataset)))
print(" ===  " + str(total_correct / len(dataloader['predict'].dataset)) + "  ===")
