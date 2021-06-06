import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


class Siamese(nn.Module):

    # Load trained weights (from supervised model)
    def load_weights_2(self, path):
        # load from pre-trained, before DistributedDataParallel constructor
        weightpath = path
        if os.path.isfile(weightpath):
            checkpoint = torch.load(weightpath, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = self.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        else:
            print("=> no checkpoint found at '{}'".format(weightpath))

    # Modify as needed depending on what needs to be frozen
    def freeze_representation(self):
        # freeze all layers but the last fc
        count = 0
        for name, param in self.conv.named_parameters():
            count += 1
        count1 = 0
        for name, param in self.conv.named_parameters():
            count1 += 1
        if (count1 < 0.9 * count):
            param.requires_grad = False

    def __init__(self, model_path):
        super(Siamese, self).__init__()
        self.conv = models.__dict__['resnet50'](pretrained=True)
        # self.load_weights()
        self.conv = torch.nn.Sequential(*(list(self.conv.children())[:-1]))
        # print((self.conv))
        self.liner = nn.Sequential(nn.Linear(2048, 512), nn.Sigmoid())
        self.out = nn.Linear(512, 1)

        self.load_weights_2(model_path)
        # self.load_weights_2('0.47_transfer_cc_50_model-inter-164001.pt')
        # Remove to disable freezing pretrained representation
        self.freeze_representation()
        # print(self.conv)

    def forward_one(self, x):
        x = self.conv(x)
        # Need to reshape explicitly, since splitting after avgpool of resnet50
        x = x.reshape(x.size(0), -1)
        # x = x.view(x.size()[0], -1)
        # x = self.liner(x)
        return x

    def forward(self, output1, output2):
        output1 = self.forward_one(output1)
        output2 = self.forward_one(output2)
        return output1, output2

    def forward_1(self, x1, x2):
        out1 = self.forward_one(x1)
        # print(out1.size())
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return torch.sigmoid(out)
        # return out

    def predict_1(self, images):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
        test_transforms1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
        new_transforms = transforms.Compose([
            normalize
        ])

        refs = images[0]

        probes = images[1]

        refs = torch.from_numpy(refs).transpose(1, 3).transpose(2, 3).float()
        probes = torch.from_numpy(probes).transpose(1, 3).transpose(2, 3).float()

        for i in range(len(probes)):
            probes[i] = new_transforms(probes[i])
        probes = probes.cuda()
        for i in range(len(refs)):
            refs[i] = new_transforms(refs[i])
        # refs = test_transforms(refs)
        refs = refs.cuda()
        # print(refs.shape)
        results = []
        return (self.forward(refs, probes)).detach().cpu().numpy()

    def predict(self, images):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
        test_transforms1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
        new_transforms = transforms.Compose([
            normalize
        ])

        refs = images[0]

        probes = images[1]

        refs = torch.from_numpy(refs).transpose(1, 3).transpose(2, 3).float()
        probes = torch.from_numpy(probes).transpose(1, 3).transpose(2, 3).float()

        for i in range(len(probes)):
            probes[i] = new_transforms(probes[i])
        probes = probes.cuda()
        for i in range(len(refs)):
            refs[i] = new_transforms(refs[i])
        # refs = test_transforms(refs)
        refs = refs.cuda()
        # print(refs.shape)
        output1, output2 = self.forward(refs, probes)
        output = F.pairwise_distance(output1, output2).detach().cpu().numpy()
        # results = (self.forward(refs, probes)).detach().cpu().numpy()

        # commented the below
        output = 1 / (1 + output)
        return output
        # print(output)
        # print(1/(1+output))
        # return (self.forward(refs, probes)).detach().cpu().numpy()


# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
