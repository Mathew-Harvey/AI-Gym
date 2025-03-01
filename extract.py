import pickle
import numpy as np
import os
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images(batch_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dict = unpickle(batch_file)
    images = data_dict[b'data']
    labels = data_dict[b'labels']
    for i, (img_data, label) in enumerate(zip(images, labels)):
        # Reshape the flat 3072 array (1024 R, 1024 G, 1024 B) to 32x32x3
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)  # HWC format
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f"image_{i}_label_{label}.jpg"))

# Process all training batches
for i in range(1, 6):
    save_images(f"data_batch_{i}", "cifar10_images/train")
save_images("test_batch", "cifar10_images/test")

# Load class names
meta = unpickle("batches.meta")
class_names = [name.decode('utf-8') for name in meta[b'label_names']]
with open("classes.txt", "w") as f:
    f.write("\n".join(class_names))