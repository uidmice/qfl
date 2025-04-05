import os, json, io
import shutil, random
from datasets import load_dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
from torch.utils.data import Subset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from collections import defaultdict

# dataset = load_dataset("benjamin-paine/imagenet-1k-64x64")
# val_dir = 'data/tiny-imagenet-200/val'
# img_dir = os.path.join(val_dir, 'images')
# ann_file = os.path.join(val_dir, 'val_annotations.txt')
# if  os.path.exists(img_dir):
# # Create class folders
#     with open(ann_file, 'r') as f:
#         for line in f:
#             img_name, class_name = line.strip().split('\t')[:2]
#             class_dir = os.path.join(val_dir, class_name)
#             if not os.path.exists(class_dir):
#                 os.makedirs(class_dir)
#             src = os.path.join(img_dir, img_name)
#             dst = os.path.join(class_dir, img_name)
#             shutil.move(src, dst)
#     # Remove old images/ directory
#     shutil.rmtree(img_dir)

# random.seed(42)
def get_or_create_selected_classes(dataset,save_path="selected_classes.json",  num_selected=50, seed=42):

    # If the selection file exists, load the selected class indices.
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            selected_classes = json.load(f)
            selected_classes = selected_classes["selected_classes"]
        print("Loaded selected classes from file.")
    else:
        # Determine total number of classes from the train split.
        total_classes = len(dataset["train"].features["label"].names)
        random.seed(seed)
        selected_classes = random.sample(range(total_classes), num_selected)
        # Save the selected classes for future use.
        with open(save_path, "w") as f:
            json.dump({"selected_classes": selected_classes}, f)
        print("Selected classes saved to file.")
        # Filter function: only include examples with labels in selected_classes.
    
    def is_selected(example):
        return example["label"] in selected_classes
    
    train_subset = dataset["train"].filter(is_selected)
    val_subset = dataset["validation"].filter(is_selected)
    
    # Remap labels to a contiguous range [0, num_selected-1]
    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(selected_classes))}
    
    def remap_label(example):
        example["label"] = label_map[example["label"]]
        return example
    
    train_subset = train_subset.map(remap_label)
    val_subset = val_subset.map(remap_label)
    
    return train_subset, val_subset, selected_classes, label_map

def get_or_create_selected_classes_dir(train_dir, save_path='selected_classes.json', num_classes=50):
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            selected_classes = json.load(f)
            print(f"Loaded {len(selected_classes)} classes from {save_path}")
    else:
        all_classes = sorted(os.listdir(train_dir))
        random.seed(42)
        selected_classes = random.sample(all_classes, num_classes)
        with open(save_path, 'w') as f:
            json.dump(selected_classes, f)
            print(f"Saved {len(selected_classes)} classes to {save_path}")
    return selected_classes



class SubclassFilter(ImageFolder):
    def __init__(self, root, classes_to_keep, transform=None):
        super().__init__(root, transform=transform)
        
        # Original to filtered index mapping
        original_to_filtered_idx = {
            self.class_to_idx[cls]: i for i, cls in enumerate(classes_to_keep)
        }

        # Filter samples
        self.samples = [
            (path, original_to_filtered_idx[label])
            for path, label in self.samples
            if label in original_to_filtered_idx
        ]
        self.targets = [label for _, label in self.samples]

        # Update class mappings
        self.classes = classes_to_keep
        self.class_to_idx = {cls: i for i, cls in enumerate(classes_to_keep)}

def random_split_clients(dataset, n_clients):
    m = len(dataset) // n_clients
    indices = np.random.permutation(len(dataset))
    client_indices = {i: indices[i * m:(i + 1) * m].tolist() for i in range(n_clients)}
    client_datasets = [ Subset(dataset, client_indices[client_id]) for client_id in range(n_clients)] 
    return client_datasets

def dirichlet_split(dataset, num_clients, num_classes, alpha=0.5):

    num_samples = len(dataset)
    
    labels = np.array([dataset[i][1] for i in range(num_samples)])
    
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    client_indices = {client: [] for client in range(num_clients)}
    
    # For each class, partition its indices among the clients.
    for c in range(num_classes):
        indices = np.array(class_indices[c])
        np.random.shuffle(indices)
        
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        counts = (proportions * len(indices)).astype(int)
        
        # Adjust counts to ensure they sum to the total number of indices.
        diff = len(indices) - counts.sum()
        for i in range(diff):
            counts[i % num_clients] += 1
        
        # Assign indices to each client.
        start = 0
        for client in range(num_clients):
            client_indices[client].extend(indices[start:start + counts[client]].tolist())
            start += counts[client]
    
    # Optionally sort indices in each client partition.
    client_indices_list = [sorted(client_indices[i]) for i in range(num_clients)]
    return client_indices_list

class MultiAugmentDataset(Dataset):
    """
    Wraps a base dataset to return multiple augmented views of each sample.
    This effectively increases the number of samples by a factor of n_views.
    """
    def __init__(self, base_dataset, n_views=2):
        self.base_dataset = base_dataset
        self.n_views = n_views

    def __len__(self):
        # Effective length is increased by n_views
        return len(self.base_dataset) * self.n_views

    def __getitem__(self, index):
        # Map index to an original sample
        orig_index = index % len(self.base_dataset)
        # Get the sample's path and label from the base dataset
        path, label = self.base_dataset.samples[orig_index]
        # Load the image using the same loader as the base dataset
        image = self.base_dataset.loader(path)
        # Apply the transform (which is stochastic) to generate an augmented view
        img = self.base_dataset.transform(image)
        return img, label

def check_samples_per_class(dataset, n_samples=3):
    """
    For each class in the dataset, randomly sample n_samples images,
    display them in a grid, and label the grid with the class name.
    """
    # Group sample indices by class label
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        subset_indices = dataset.indices
    else:
        base_dataset = dataset
        subset_indices = list(range(len(dataset)))

    
    # Group indices by class label using the base_dataset's samples attribute.
    label_to_indices = {}
    for idx in subset_indices:
        _, label = base_dataset.samples[idx]
        label_to_indices.setdefault(label, []).append(idx)
    
    # For each class, sample images and display them.
    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        sample_indices = random.sample(indices, min(n_samples, len(indices)))
        # Use base_dataset to access the sample (images, label) tuple.
        images = [base_dataset[i][0] for i in sample_indices]
        
        grid_img = make_grid(images, nrow=n_samples, normalize=True)
        plt.figure(figsize=(n_samples * 2, 2))
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Class {label}: {base_dataset.classes[label]}")
        plt.axis('off')
        plt.show()

# Paths
# train_dir = 'data/tiny-imagenet-200/train'
# val_dir = 'data/tiny-imagenet-200/val'
# class_file = 'data/tiny-imagenet-200/selected_classes.json'

# # Get selected classes
# selected_classes = get_or_create_selected_classes(train_dir, class_file)

# train_transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.RandomCrop(64, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     # Optionally add normalization with Tiny ImageNet stats:
#     transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282]),
# ])

# # Use a simpler transform for validation (no randomness)
# val_transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282]),
# ])


# # Datasets
# train_ds = SubclassFilter(train_dir, selected_classes, train_transform)
# test_ds = SubclassFilter(val_dir, selected_classes, val_transform)   
# augmented_train_ds = MultiAugmentDataset(train_ds, n_views=3)  # For example, 3 augmented views per image

# print(f"Filtered dataset contains {len(train_ds)} images.")
# print(f"Filtered validation dataset contains {len(test_ds)} images.")

# # Parameters for client splitting
# n_clients = 10   # Number of clients
# m = 1000        # Number of data points per client

# # Randomly split the dataset into n_clients, each with m data points
# client_datasets = random_split_clients(train_ds, n_clients, m)


# Visualize some sample images from the training and validation datasets
# check_samples_per_class(client_datasets[0])

# check_samples_per_class(test_ds)