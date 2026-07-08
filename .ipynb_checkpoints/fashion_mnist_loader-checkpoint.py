import numpy as np
import struct
import os
import matplotlib.pyplot as plt

def load_mnist_images(file_path):
    """
    Load MNIST or Fashion MNIST images from a ubyte file.
    Args:
        file_path (str): Path to the images ubyte file.
    Returns:
        numpy.ndarray: A 2D array of shape (num_images, 28*28) containing the image data.
    """
    with open(file_path, 'rb') as f:
        # Read the magic number and number of images
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file: {file_path}")
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)  # Reshape to (num_images, 28, 28)
        images = images.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    return images

def load_mnist_labels(file_path):
    """
    Load MNIST or Fashion MNIST labels from a ubyte file.
    Args:
        file_path (str): Path to the labels ubyte file.
    Returns:
        numpy.ndarray: A 1D array of shape (num_labels,) containing the labels.
    """
    with open(file_path, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file: {file_path}")
        
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def visualize_samples(images, labels, class_names, num_samples=25):
    """
    Visualize a grid of sample images with their labels.
    Args:
        images (numpy.ndarray): Array of images of shape (num_images, 28, 28).
        labels (numpy.ndarray): Array of labels of shape (num_labels,).
        class_names (list): List of class names corresponding to label indices.
        num_samples (int): Number of samples to display (must be a perfect square).
    """
    grid_size = int(np.sqrt(num_samples))  # Determine grid size
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle("Fashion MNIST Samples", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        ax.imshow(images[i], cmap='gray')  # Display the image
        ax.set_title(class_names[labels[i]])  # Display the label
        ax.axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust space for the title
    plt.show()

# Paths to the ubyte files (update these paths to match your file locations)
train_images_path = "train-images-idx3-ubyte"
train_labels_path = "train-labels-idx1-ubyte"
test_images_path = "t10k-images-idx3-ubyte"
test_labels_path = "t10k-labels-idx1-ubyte"

# Load the data
if os.path.exists(train_images_path) and os.path.exists(train_labels_path):
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    print(f"Loaded {train_images.shape[0]} training images and {train_labels.shape[0]} labels.")
else:
    print("Training data files not found. Please check the file paths.")

if os.path.exists(test_images_path) and os.path.exists(test_labels_path):
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)
    print(f"Loaded {test_images.shape[0]} test images and {test_labels.shape[0]} labels.")
else:
    print("Test data files not found. Please check the file paths.")

# Class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Visualize a grid of sample images
visualize_samples(train_images, train_labels, class_names, num_samples=25)