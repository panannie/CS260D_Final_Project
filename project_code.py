# -*- coding: utf-8 -*-
"""
From Google Colab
"""

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np

#add spurious correlations to MNIST
class SpuriousMNIST(MNIST): #with init
    def __init__(self, *args, spurious_strength=255, **kwargs):
        super().__init__(*args, **kwargs)
        self.spurious_strength = spurious_strength

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = np.array(img)

        #white box for labels < 5
        if label < 5:
            img[:5, :5] = self.spurious_strength

        img = transforms.ToTensor()(img)
        return img, label, index

train_dataset = SpuriousMNIST(
    root='./data', train=True, download=True,
    transform=None
)

import numpy as np

#forgettability tracker
class ForgettabilityTracker:
    def __init__(self, num_samples):
        self.forget_counts = np.zeros(num_samples)

    def update(self, predictions, labels, indices):
        incorrect = (predictions != labels).cpu().numpy()
        self.forget_counts[indices] += incorrect

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),  #hidden layer
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

    def get_features(self, x):
        return self.encoder(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.mean(torch.relu(pos_dist - neg_dist + self.margin))
        return loss

#hybrid loss with cross-entropy and contrastive loss
class HybridLoss:
    def __init__(self, ce_weight=1.0, contrastive_weight=1.0):
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss()

    def __call__(self, model, data, labels, pairs=None):
        outputs = model(data)
        classification_loss = self.cross_entropy_loss(outputs, labels)

        if pairs:
            anchor, positive, negative = pairs
            features = model.get_features(data)
            contrastive_loss = self.contrastive_loss(features[anchor], features[positive], features[negative])
            return classification_loss + self.contrastive_weight * contrastive_loss
        else:
            return classification_loss

input_dim = 28 * 28
hidden_dim = 128
output_dim = 10
model = SimpleModel(input_dim, hidden_dim, output_dim)

hybrid_loss_fn = HybridLoss(ce_weight=1.5, contrastive_weight=1.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)  #learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  #lr scheduler

#metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

#testing
def sample_pairs(data, labels, forget_counts):
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and labels[i] == labels[j]:
                if forget_counts[i] != forget_counts[j]:  #spurious correlation flag?
                    pairs.append((i, j, np.random.choice(np.where(labels != labels[i])[0])))
    return pairs

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

#MNIST
test_dataset = SpuriousMNIST(
    root='./data', train=False, download=True,
    transform=None
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

features, labels = [], []
model.eval()
with torch.no_grad():
    for data, label, _ in test_loader:
        outputs = model(data)
        features.append(outputs)
        labels.append(label)

features = torch.cat(features).cpu().numpy()
labels = torch.cat(labels).cpu().numpy()

tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(features)

#t-SNE results
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='jet')
plt.colorbar()
plt.title("Feature Representations via t-SNE")
plt.show()

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

#test dataset
test_dataset = SpuriousMNIST(
    root='./data', train=False, download=True,
    transform=None  # Custom transform is handled in the class
)

#test dataloader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

features, labels = [], []
model.eval()
with torch.no_grad():
    for data, label, _ in test_loader:
        outputs = model(data)
        features.append(outputs)
        labels.append(label)

features = torch.cat(features).cpu().numpy()
labels = torch.cat(labels).cpu().numpy()

#t-SNE
tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(features)

plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='jet')
plt.colorbar()
plt.title("Feature Representations via t-SNE")
plt.show()

#WORKING DRAFT
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

#add spurious correlations to MNIST
class SpuriousMNIST(MNIST): #without init
    def __getitem__(self, index):
        img, label = super().__getitem__(index)  #img is PIL
        img = np.array(img)

        #spurious white box for labels 0-4
        if label < 5:
            img[:5, :5] = 255

        img = transforms.ToTensor()(img)
        return img, label, index

#train dataset
train_dataset = SpuriousMNIST(
    root='./data', train=True, download=True,
    transform=None
)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

#train dataloader
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

model = SimpleModel(input_dim=28*28, hidden_dim=128, output_dim=10)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
forget_tracker = defaultdict(int)  #track forgettability per sample

#training parameters and tracking
num_epochs = 50
train_accuracies, val_accuracies = [], []
train_losses, val_losses = [], []
forgettability_per_epoch = []

#training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for data, labels, indices in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = hybrid_loss_fn(model, data, labels, pairs=None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels)

        for idx, is_correct in zip(indices, correct):
            if not is_correct:  #increment count if misclassified
                forget_tracker[idx.item()] += 1

        correct_train += correct.sum().item()
        total_train += labels.size(0)

    scheduler.step()
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss / len(train_loader))

    epoch_forgettability = list(forget_tracker.values())
    forgettability_per_epoch.append(epoch_forgettability)

    #validation
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data, labels, _ in val_loader:
            outputs = model(data)
            loss = hybrid_loss_fn(model, data, labels, pairs=None)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

#forgettability distribution every 5 epochs
plt.figure(figsize=(10, 6))
for epoch, scores in enumerate(forgettability_per_epoch):
    if (epoch + 1) % 5 == 0:
        plt.hist(scores, bins=20, alpha=0.5, label=f'Epoch {epoch+1}')
#logarithmic scale y-axis
ax = plt.gca()
ax.set_yscale('log')
plt.title('Forgettability Score Distribution Over Epochs (Every 5 Epochs, Log Scale)')
plt.xlabel('Forgettability Score')
plt.ylabel('Frequency (Log Scale)')
plt.legend()
plt.show()

#loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='o')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Val Accuracy', marker='o')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

#FINAL
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

class SpuriousMNIST(MNIST):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = np.array(img)

        #spurious white box for labels 0-4
        if label < 5:
            img[0:5, 0:5] = 255

        img = transforms.ToTensor()(img)
        return img, label, index

class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

#hybrid loss function
def hybrid_loss_fn(model, data, labels, pairs=None, contrastive_loss_fn=None):
    classification_loss = torch.nn.CrossEntropyLoss()(model(data), labels)

    if pairs:
        features = model.get_features(data)
        #contrastive loss for each pair (anchor, positive, negative)
        contrastive_loss = 0
        for anchor_idx, positive_idx, negative_idx in pairs:
            contrastive_loss += contrastive_loss_fn(features[anchor_idx], features[positive_idx], features[negative_idx])
        contrastive_loss /= len(pairs)
        return classification_loss + contrastive_loss
    else:
        return classification_loss

def create_pairs(data, labels):
    pairs = []
    for i, anchor_label in enumerate(labels):
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
        positive_index = positive_indices[random.choice(range(len(positive_indices)))]

        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
        negative_index = negative_indices[random.choice(range(len(negative_indices)))]

        pairs.append((i, positive_index.item(), negative_index.item()))
    return pairs

train_dataset = SpuriousMNIST(
    root='./data', train=True, download=True, transform=None
)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

#dataloaders
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

model = SimpleModel(input_dim=28*28, hidden_dim=128, output_dim=10)
optimizer = optim.Adam(model.parameters(), lr=0.0001) #learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

#forgettability tracker
forget_tracker = defaultdict(lambda: 0)
for idx in range(len(train_subset)):
    forget_tracker[idx] = 0

#training parameters and tracking
num_epochs = 50
train_accuracies, val_accuracies = [], []
train_losses, val_losses = [], []
forgettability_per_epoch = []

#accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    return (predicted == labels).sum().item(), predicted

#training and val loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for data, labels, indices in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        pairs = create_pairs(data, labels)
        loss = hybrid_loss_fn(model, data, labels, pairs=pairs, contrastive_loss_fn=contrastive_loss_fn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        correct, predicted = calculate_accuracy(outputs, labels)
        correct_train += correct
        total_train += labels.size(0)

        #update forgettability count
        for idx, is_correct in zip(indices, (predicted == labels)):
            if not is_correct:
                forget_tracker[idx.item()] += 1

    scheduler.step()
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss / len(train_loader))

    epoch_forgettability = list(forget_tracker.values())
    forgettability_per_epoch.append(epoch_forgettability)

    #validation
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data, labels, _ in val_loader:
            outputs = model(data)
            loss = hybrid_loss_fn(model, data, labels)
            val_loss += loss.item()
            correct_val += calculate_accuracy(outputs, labels)[0]
            total_val += labels.size(0)

    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

#forgettability score distribution (every 5 epochs)
plt.figure(figsize=(10, 6))
for epoch, scores in enumerate(forgettability_per_epoch):
    if (epoch + 1) % 5 == 0:
        plt.hist(scores, bins=20, alpha=0.5, label=f'Epoch {epoch+1}')
plt.yscale('log') #log scale
plt.title('Forgettability Score Distribution (Log Scale)')
plt.xlabel('Forgettability Score')
plt.ylabel('Frequency (Log Scale)')
plt.legend()
plt.show()

#loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Val Loss', marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', marker='o')
plt.plot(val_accuracies, label='Val Accuracy', marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#cleaned up forgettability
import matplotlib.pyplot as plt

#forgettability distribution every 5 epochs
plt.figure(figsize=(10, 6))

for epoch, scores in enumerate(forgettability_per_epoch):
    if (epoch + 1) % 5 == 0:
        plt.hist(scores, bins=20, alpha=0.5, label=f'Epoch {epoch+1}')

#logarithmic scale y-axis
ax = plt.gca()
ax.set_yscale('log')

plt.title('Forgettability Score Distribution Over Epochs (Every 5 Epochs, Log Scale)')
plt.xlabel('Forgettability Score')
plt.ylabel('Frequency (Log Scale)')
plt.legend()
plt.show()

#visual set up
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

class SpuriousMNIST(MNIST):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = transforms.ToPILImage()(img)
        img = np.array(img)

        #spurious feature
        if label < 5:
            img[0:5, 0:5] = 255

        img = transforms.ToTensor()(img)
        return img, label, index

spurious_mnist = SpuriousMNIST(
    root='./data', train=True, download=True,
    transform=transforms.ToTensor()
)

#visual for presentation and paper
def display_spurious_and_clean(dataset, index):
    spurious_img, label, _ = dataset[index]
    #clean image
    clean_img, _ = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())[index]
    spurious_img = spurious_img.squeeze().numpy()
    clean_img = clean_img.squeeze().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    #spurious image
    ax1.imshow(spurious_img, cmap='gray')
    ax1.set_title(f"Spurious Image (Label: {label})")
    ax1.axis('off')
    #clean image
    ax2.imshow(clean_img, cmap='gray')
    ax2.set_title(f"Clean Image (Label: {label})")
    ax2.axis('off')
    plt.show()

display_spurious_and_clean(spurious_mnist, index=10)
