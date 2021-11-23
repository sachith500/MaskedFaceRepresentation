import json

from torch.utils.data import DataLoader
from torchvision import datasets

from AJCAI2021.accuracy_calculator import MAECalculator, PercentageCalculator
from AJCAI2021.model import *


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class Pipeline:

    def __init__(self, type='age_classification', batch_size=1):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        model_path = self.config["models"][type]["model_path"]
        accuracy_type = self.config["models"][type]["model_path"]
        dataset_folder = self.config["dataset_path"]

        self.model = None
        self.data_loader = None
        self.accuracy_calculator = None

        self.dataset_folder = dataset_folder
        self.shape = 224
        self.batch_size = batch_size
        self.type = type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model(model_path)
        self.build_dataset()
        self.build_accuracy_calculator(accuracy_type)

    def build_accuracy_calculator(self, accuracy_type):
        if accuracy_type == "mae":
            self.accuracy_calculator = MAECalculator()
        elif accuracy_type == "percentage":
            self.accuracy_calculator = PercentageCalculator()
        else:
            self.accuracy_calculator = PercentageCalculator()

    def build_model(self, model_path):
        if self.type == 'age_classification':
            self.model = AgeClassificationNN(model_path)
        elif self.type == 'sex':
            self.model = SexClassificationNN(model_path)
        elif self.type == 'race':
            self.model = RaceClassificationNN(model_path)
        else:
            self.model = AgeRegressionNN(model_path)

        self.model.load_weights()

    def build_dataset(self):
        transform = self.get_transforms(self.shape)
        dataset = datasets.ImageFolder(self.dataset_folder, transform=transform)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def get_transforms(self, shape=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        transform = torchvision.transforms.Compose([
            # torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(shape),
            torchvision.transforms.CenterCrop(shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])

        return transform

    def process(self):
        data_comparison = []
        for inputs, labels in self.data_loader:
            inputs = inputs.to(self.device)
            output_labels = self.model(inputs)
            labels = labels.numpy()
            batch_output = [output_labels, labels]
            data_comparison.append(batch_output)

        self.accuracy_calculator.calculate(data_comparison)
