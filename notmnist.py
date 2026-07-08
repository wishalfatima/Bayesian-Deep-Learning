import numpy as np
import struct
import matplotlib.pyplot as plt

def load_ubyte_images(file_path):
    """
    Load images from a ubyte file.
    Args:
        file_path (str): Path to the images ubyte file.
    Returns:
        numpy.ndarray: A 3D array of shape (num_images, 28, 28) containing the image data.
    """
    with open(file_path, 'rb') as f:
        # Read the magic number, number of images, rows, and columns
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file: {file_path}")
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)  # Reshape to (num_images, 28, 28)
    return images

def load_ubyte_labels(file_path):
    """
    Load labels from a ubyte file.
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

def visualize_notmnist_samples(images, labels, class_names, num_samples=25):
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
    fig.suptitle("notMNIST Samples", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        ax.imshow(images[i], cmap='gray')  # Display the image
        ax.set_title(class_names[labels[i]])  # Display the label
        ax.axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust space for the title
    plt.show()

# Paths to the ubyte files
image_file_path = "notmnist-images-idx3-ubyte"
label_file_path = "notmnist-labels-idx1-ubyte"

# Load the data
print("Loading notMNIST data...")
images = load_ubyte_images(image_file_path)
labels = load_ubyte_labels(label_file_path)
print(f"Loaded {images.shape[0]} images and {labels.shape[0]} labels.")

# Class names for notMNIST (A-J)
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Visualize a grid of sample images
visualize_notmnist_samples(images, labels, class_names, num_samples=25)