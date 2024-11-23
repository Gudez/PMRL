import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
import torch.nn as nn
from torchsummary import summary

### Point 2: Load Traffic sign data for deep neural network processing
########################################################################
def train_val_dataset(dataset, val_split=0.2):
    """
    Split the dataset into training & validation test
    :param dataset: to be splitted
    :param val_split: proportion for the validation dataset (default 20%)
    :return:
    """
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = torch.utils.data.Subset(dataset, train_idx)
    datasets['validation'] = torch.utils.data.Subset(dataset, val_idx)
    return datasets

# Datasets & DataLoaders in PyTorch
class SignDataset(torch.utils.data.Dataset):
    """
    Class Dataset: it stores the samples and their corresponding labels.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.classes = []

        # Get the class names (subfolder names)
        self.classes = sorted(os.listdir(root_dir))

        # Iterate over the classes (subfolders)
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                # Iterate over the image files in each class directory
                for image_file in os.listdir(class_dir):
                    if image_file.endswith('.jpg'):
                        self.image_files.append(os.path.join(class_dir, image_file))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        # 1 if "snow", 0 if "twenty"
        label = 1 if 'class1' in image_path else 0
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# Define paths to source folders
source_folder = 'C:/Users/ricca/PycharmProjects/MachineLearning/GTSRB_subset_2'

# Define the transformation
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0)), # It's already 64x64
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Widely used values from ImageNet
])

# Create an instance of the SignDataset class
sign_dataset = SignDataset(root_dir=source_folder, transform=transform)

# Split the class dataset into train and validation datasets
sign_datasets = train_val_dataset(sign_dataset)

# Create a DataLoader to load the dataset in batches (in total 32)
# It wraps an iterable around the Dataset to enable easy access to the samples
# In total 660 elements, therefore I expect 33 batches of 32 files + rest
# (splitted between training and validation)
dataloaders = {x:torch.utils.data.DataLoader(sign_datasets[x], batch_size=32, shuffle=True) for x in ['train','validation']}
train_features, train_labels = next(iter(dataloaders['train']))

### Point 5: Implement the CNN class in PyTorch
#################################

# DEFINE AND INITIALIZE THE NEURAL NETWORK
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 1st convolutional layer: 10 filters 3x3 with stride 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10,
                               kernel_size=(3, 3), stride=2)
        # Activation function
        self.act = nn.ReLU()
        # 2x2 max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # 2nd convolutional layer: 10 filters 3x3 with stride 2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10,
                               kernel_size=(3, 3), stride=2)
        #  Flattening + full-connected (dense) layer of 2 neurons
        self.flatten = nn.Flatten(1, -1)
        self.dense1 = nn.Linear(in_features=10 * 3 * 3, out_features=2)
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    # x represents our data
    def forward(self, x):
        # First layer
        x = self.conv1(x)
        # Activation
        x = self.act(x)
        # Max pooling
        x = self.maxpool(x)
        # Second layer: repeat
        x = self.conv2(x)
        x = self.act(x)
        x = self.maxpool(x)
        # Flattening
        x = self.flatten(x)
        # Dense layer
        x = self.dense1(x)
        # Output
        x = self.sigmoid(x)
        return x

conv_model = ConvNet()
print(conv_model)

# Print the summary of the model
summary(conv_model,(3, 64, 64))

num_epochs = 20
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(conv_model.parameters(), lr=0.06)

# Training loop
print("Loss function after each epoch")
for n in range(num_epochs):
    loss_e = 0
    for tr_images, tr_labels in dataloaders['train']:
        y_pred = conv_model(tr_images)
        loss = loss_fn(y_pred.float(), tr_labels)
        loss_e += loss.item() * tr_images.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(np.round(loss_e / len(dataloaders['train'].sampler), 3))

misclassified = 0
total = len(dataloaders['validation'].sampler)

# Set the model to evaluation mode
conv_model.eval()

# Evaluation of the model
with torch.no_grad():
    for val_images, target in dataloaders['validation']:
        test_labels_onehot_p = conv_model(val_images.float())
        test_labels_p = np.argmax(test_labels_onehot_p.detach().numpy(),axis=1)
        misclassified += np.count_nonzero(target-test_labels_p)

accuracy = 1 - misclassified / total
print('Accuracy on validation set: {:.2f}%'.format(100 * accuracy))