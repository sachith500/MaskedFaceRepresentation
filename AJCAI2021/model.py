import os

import torch
import torch.nn as nn
import torchvision


class ClassificationNetwork(nn.Module):

    def __init__(self):
        super(ClassificationNetwork, self).__init__()
        self.model_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def forward(self, inputs):
        return self.model(inputs)

    def build_model(self, architecture, num_classes):
        print(architecture)
        model = torchvision.models.__dict__[architecture](pretrained=False)
        model.fc = torch.nn.Linear(2048, num_classes)
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        return model

    def load_weights(self):
        weight_path = self.model_path
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    new_name = f"model.{k[len('module.'):]}"
                    state_dict[new_name] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            self.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.model.to(self.device)


class RegressionNetwork(nn.Module):

    def __init__(self, m=60, c=60):
        super(RegressionNetwork, self).__init__()
        self.model_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model('resnet50', self.no_of_classes)
        self.m = m
        self.c = c

    def forward(self, inputs):
        output = self.model(inputs)
        output = output.to(self.device)
        out_labels = self.m * output + self.c  # regression for age
        out_labels = torch.transpose(out_labels, 0, 1).to('cpu').detach().numpy()
        return out_labels

    def build_model(self, architecture, num_classes):
        model = torchvision.models.__dict__[architecture](pretrained=False)
        model.fc = torch.nn.Linear(2048, num_classes)
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        return model

    def load_weights(self):
        weight_path = self.model_path
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    new_name = f"model.{k[len('module.'):]}"
                    state_dict[new_name] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            self.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.model.to(self.device)


class AgeClassificationNN(ClassificationNetwork):

    def __init__(self, model_path):
        super(AgeClassificationNN, self).__init__()
        self.model_path = model_path
        self.no_of_classes = 7
        self.model = self.build_model('resnet50', self.no_of_classes)

    def forward(self, inputs):
        output = self.model(inputs)
        output = output.to(self.device)
        out_labels = torch.argmax(output, dim=1)
        out_labels = out_labels.to('cpu')
        out_labels = out_labels.detach().numpy()
        # out_labels = output * 60 + 60  # regression for age
        # out_labels = torch.transpose(out_labels, 0, 1).to('cpu')
        return out_labels

    def build_model(self, architecture, num_classes):
        model = torchvision.models.__dict__[architecture](pretrained=False)
        model.fc = torch.nn.Linear(2048, num_classes)
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        return model


class AgeRegressionNN(ClassificationNetwork):

    def __init__(self, model_path):
        super(AgeRegressionNN, self).__init__()
        self.model_path = model_path
        self.no_of_classes = 1
        self.model = self.build_model('resnet50', self.no_of_classes)

    def forward(self, inputs):
        output = self.model(inputs)
        output = output.to(self.device)
        out_labels = output * 60 + 60  # regression for age
        out_labels = torch.transpose(out_labels, 0, 1).to('cpu').detach().numpy()
        return out_labels

    def build_model(self, architecture, num_classes):
        model = torchvision.models.__dict__[architecture](pretrained=False)
        model.fc = torch.nn.Linear(2048, num_classes)
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        return model


class RaceClassificationNN(ClassificationNetwork):

    def __init__(self, model_path):
        self.no_of_classes = 5  # 0: "White",1: "Black", 2: "Asian", 3: "Indian", 4: "Others"
        super(RaceClassificationNN, self).__init__()
        self.model_path = model_path
        self.model = self.build_model('resnet50', self.no_of_classes)

    def forward(self, inputs):
        output = self.model(inputs)
        output = output.to(self.device)
        out_labels = torch.argmax(output, dim=1)
        out_labels = out_labels.to('cpu')

        return out_labels


class SexClassificationNN(ClassificationNetwork):

    def __init__(self, model_path):
        self.no_of_classes = 2  # 0 : male, 1 : female
        super(SexClassificationNN, self).__init__()
        self.model_path = model_path
        self.model = self.build_model('resnet50', self.no_of_classes)

    def forward(self, inputs):
        output = self.model(inputs)
        output = output.to(self.device)
        out_labels = torch.argmax(output, dim=1)
        out_labels = out_labels.to('cpu')
        out_labels = out_labels.detach().numpy()

        return out_labels

    def build_model(self, architecture, num_classes):
        model = torchvision.models.__dict__[architecture](pretrained=False)
        model.fc = torch.nn.Linear(2048, num_classes)
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        return model
