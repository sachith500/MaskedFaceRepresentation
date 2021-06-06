import abc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


class SiameseNetwork(nn.Module):

    def __init__(self, model_path):
        super(SiameseNetwork, self).__init__()
        self.conv = models.__dict__['resnet50'](pretrained=True)
        self.conv = torch.nn.Sequential(*(list(self.conv.children())[:-1]))
        self.liner = nn.Sequential(nn.Linear(2048, 512), nn.Sigmoid())
        self.out = nn.Linear(512, 1)

        self.load_weights(model_path)
        self.freeze_representation()

    def load_weights(self, path):
        weight_path = path
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            self.load_state_dict(state_dict, strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(weight_path))

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

    @abc.abstractmethod
    def forward(self, output1, output2):
        pass

    @abc.abstractmethod
    def predict(self, images):
        pass

    @staticmethod
    def load_data_to_gpu(images):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_transformation = transforms.Compose([
            normalize
        ])

        refs = images[0]
        probes = images[1]

        refs = torch.from_numpy(refs).transpose(1, 3).transpose(2, 3).float()
        probes = torch.from_numpy(probes).transpose(1, 3).transpose(2, 3).float()

        for i in range(len(refs)):
            refs[i] = image_transformation(refs[i])

        for i in range(len(probes)):
            probes[i] = image_transformation(probes[i])

        refs = refs.cuda()
        probes = probes.cuda()

        return refs, probes

    def forward_sister_network(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return x


class SiameseNetworkWithSigmoid(SiameseNetwork):

    def __init__(self, model_path):
        super(SiameseNetworkWithSigmoid, self).__init__(model_path)

    def forward(self, x1, x2):
        out1 = self.forward_sister_network(x1)
        out1 = self.liner(out1)
        out2 = self.forward_sister_network(x2)
        out2 = self.liner(out2)

        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return torch.sigmoid(out)

    def predict(self, images):
        refs, probes = self.load_data_to_gpu(images)
        output = (self.forward(refs, probes)).detach().cpu().numpy()

        return output


class SiameseNetworkWith2048Distance(SiameseNetwork):

    def __init__(self, model_path):
        super(SiameseNetworkWith2048Distance, self).__init__(model_path)

    def forward(self, output1, output2):
        output1 = self.forward_sister_network(output1)
        output2 = self.forward_sister_network(output2)
        return output1, output2

    def predict(self, images):
        refs, probes = self.load_data_to_gpu(images)
        output1, output2 = self.forward(refs, probes)
        output = F.pairwise_distance(output1, output2).detach().cpu().numpy()

        output = 1 / (1 + output)
        return output


class SiameseNetworkWith512Distance(SiameseNetwork):

    def __init__(self, model_path):
        super(SiameseNetworkWith512Distance, self).__init__(model_path)

    def forward(self, output1, output2):
        out1 = self.forward_sister_network(output1)
        output1 = self.liner(out1)
        out2 = self.forward_sister_network(output2)
        output2 = self.liner(out2)
        return output1, output2

    def predict(self, images):
        refs, probes = self.load_data_to_gpu(images)
        output1, output2 = self.forward(refs, probes)
        output = F.pairwise_distance(output1, output2).detach().cpu().numpy()

        output = 1 / (1 + output)
        return output
