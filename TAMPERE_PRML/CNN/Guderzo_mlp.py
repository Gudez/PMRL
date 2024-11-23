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
        # self.images_paths = glob(f'{root_dir}/*/*')
        # self.images_pythfiles = [i for i in self.images_paths if i.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        # 1 if "snow", 0 if "twenty"
        label = 1 if 'class1' in image_path else 0
        # label = self.labels[idx]
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

## Size of tensor (32 images, each with 3 channels (RGB)
## and each image having dimensions of 64x64 pixels)
# print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")

# Loop through the DataLoader to get a batch on each loop
# Each iteration returns a batch of data, where the batch size is determined
# by the parameter specified during initialization.

## Sanity check
## Training set
#for images, labels in dataloaders['train']:
#    print(images.shape, labels) # Size([batch, channels, pixels])
#    #print(images.shape[0])
#    pass

## Validation set
#for images, labels in dataloaders['validation']:
#    print(images.shape, labels) # Size([batch, channels, pixels])
#    #print(images.shape[0])
#    pass

### Point 3: Define the network
#################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Flattening
        self.flatten = nn.Flatten()
        # 1st layer there are 100 nodes
        self.dense1 = nn.Linear(in_features=3 * 64 * 64, out_features=100)
        # 2nd layer receives the output of 100 neurons, there are 100 nodes again
        self.dense2 = nn.Linear(in_features=100, out_features=100)
        # Output layer
        self.dense3 = nn.Linear(in_features=100, out_features=2)
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    # x represents our data
    def forward(self, x):
        # Flatten the matrix
        x = self.flatten(x)
        # Dense layer 1
        x = self.dense1(x)
        # Dense layer 2
        x = self.dense2(x)
        # Final layer
        x = self.dense3(x)
        # Output
        x = self.sigmoid(x)
        return x

model = Net()
print(model)

# Print the summary of the model
summary(model,(3, 64, 64))

num_epochs = 10
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Training loop
print("Loss function after each epoch")
for n in range(num_epochs):
    loss_e = 0
    for tr_images, tr_labels in dataloaders['train']:
        y_pred = model(tr_images)
        loss = loss_fn(y_pred.float(), tr_labels)
        loss_e += loss.item() * tr_images.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(np.round(loss_e / len(dataloaders['train'].sampler), 3))

misclassified = 0
total = len(dataloaders['validation'].sampler)

# Set the model to evaluation mode
model.eval()

# Evaluation of the model
with torch.no_grad():
    for val_images, target in dataloaders['validation']:
        test_labels_onehot_p = model(val_images.float())
        test_labels_p = np.argmax(test_labels_onehot_p.detach().numpy(),axis=1)
        misclassified += np.count_nonzero(target-test_labels_p)

accuracy = 1 - misclassified / total
print('Accuracy on validation set: {:.2f}%'.format(100 * accuracy))