import cv2
import os
import torch
from torch import nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(2000)

batch_size = 30
image_folders = os.listdir("Dataset/Images")
device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Device: {device}")

# Splits and distributions for generating batches
num_types = len(image_folders)
uniform_types_distribution = torch.ones(num_types, dtype = torch.float32, device = device) / num_types # 5 types of rice

# 5 types of rice, 15000 images for each = 75000 total
# 75000 * 0.2 = 15000 images in total for test/validation split
# 15000 / 5 = 3000 images of each type for test/validation split
test_split_multiplier = 0.2
num_test_imgs = int((75000 * test_split_multiplier) / 5)
num_train_imgs = int(15000 - num_test_imgs)
print(num_test_imgs, num_train_imgs)

uniform_train_images_distribution = torch.ones(num_train_imgs, dtype = torch.float32, device = device) / num_train_imgs # 15000 images for each type of rice
uniform_test_images_distribution = torch.ones(num_test_imgs, dtype = torch.float32, device = device) / num_test_imgs

print(uniform_train_images_distribution.dtype)
print(uniform_types_distribution.dtype)
print(uniform_types_distribution)
print(uniform_train_images_distribution)

def img_to_matrix(image_indexes, r_type_indexes):

    # Creates all pixel matrixes in a batch

    # Find all rice types in the selected batch
    rice_types = [image_folders[r_type_i] for r_type_i in r_type_indexes]

    # print(rice_types)
    # print(image_indexes)
    
    matrices = []

    for img_type, img_num in zip(rice_types, image_indexes):

        # img_num.item() as img_num is a tensor containing the image number
        img_path = f"Dataset/Images/{img_type}/{img_type} ({str(img_num.item())}).jpg" # E.g. Dataset/Images/Jasmine/Jasmine (3).jpg"

        # Read image into numpy array
        img = cv2.imread(img_path)

        # Scale down image
        img = cv2.resize(img, (100, 100))

        # Convert image to grey
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Matrix of shape (62500) with each pixel containing the greyscale value
        pixel_matrix = torch.from_numpy(grey_img).view(100 * 100).float()
        
        # Add to matrices list
        matrices.append(pixel_matrix)

        # Release image memory
        del img

    # Add all the matrices into a single tensor [Shape: (batch_size, pixel_number)]
    matrices = torch.stack(matrices, dim = 0)
    
    return matrices
    
def generate_batch(batch_size, split):
    
    # Generate indexes for rice types
    rice_type_indexes = torch.multinomial(input = uniform_types_distribution, num_samples = batch_size, replacement = True)

    # Convert the index into a one-hot vector for the entire batch
    rice_types = torch.zeros(batch_size, 5, device = device)
    rice_types[torch.arange(batch_size), rice_type_indexes - 1] += 1 # 2nd option = index 1 (Possible indexes = 0, 1, 2, 3, 4 ... num_types)
    
    # -------------------------------

    # Test split images
    if split == "Test":
        # Note: uniform_test_images_distribution will have a smaller range of index values 
        # - (e.g. if num_test_imgs = 15000, as there are 5 types, the last 15000 / 3000 images are for the test/validation split)
        # - i.e. Only images at index 12000 -> 15000 will be used for the test, whereas the 0 -> 12000 will be used for the train split
        rice_image_indexes = torch.multinomial(input = uniform_test_images_distribution, num_samples = batch_size, replacement = True)
        rice_image_indexes += (15000 - num_test_imgs) + 1 # The indexes only go from 0 - 14999 but the numbers at the end of each image go from 1 - 15000

    
    # Train split images
    else:
        # Generate indexes for rice images
        rice_image_indexes = torch.multinomial(input = uniform_train_images_distribution, num_samples = batch_size, replacement = True)
        rice_image_indexes += 1 # The indexes only go from 0 - 14999 but the numbers at the end of each image go from 1 - 15000
    
    # Convert indexes to matrices
    rice_image_matrices = img_to_matrix(image_indexes = rice_image_indexes, r_type_indexes = rice_type_indexes)

    # Pixel matrices, Labels
    return rice_image_matrices.to(device = device), rice_types.to(device = device)
        

# Shape = (batch_size, number of pixels in each image, 1) (batch_size of 62500 pixels which contain a greyscale value)
# Xtr, Ytr = generate_batch(batch_size = batch_size, split = "Train")
# Xte, Yte = generate_batch(batch_size = batch_size, split = "Test")

# print(Xtr.shape, Ytr.shape)
# print(Xte.shape, Yte.shape)


# No.of inputs = Number of pixels in image 

model = nn.Sequential(
                        nn.Linear(10000, 5000),
                        nn.BatchNorm1d(num_features = 5000),
                        nn.ReLU(),

                        nn.Linear(5000, 2500),
                        nn.BatchNorm1d(num_features = 2500),
                        nn.ReLU(),

                        nn.Linear(2500, 1250),
                        nn.BatchNorm1d(num_features = 1250),
                        nn.ReLU(),

                        nn.Linear(1250, 625),
                        nn.BatchNorm1d(num_features = 625),
                        nn.ReLU(),

                        nn.Linear(625, 5),

                        # nn.Linear(5000, 5)
                        )
model.to(device = device)

# Optimisers
# optimiser = torch.optim.SGD(model.parameters(), lr = 0.1) # Stochastic gradient descent
optimiser = torch.optim.AdamW(model.parameters(), lr = 1e-5) # Adam (updates learning rate for each weight individually)

Xtr, Ytr = generate_batch(batch_size = batch_size, split = "Train")
print(Xtr.shape)

losses_i = []

for i in range(1000):
    
    # Generate batch of images
    Xtr, Ytr = generate_batch(batch_size = batch_size, split = "Train")

    # Forward pass
    logits = model(Xtr) # (batch_size, 5)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, Ytr)
    
    # Set gradients to 0
    optimiser.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update model parameters
    optimiser.step()
    
    # -----------------------------------------------
    # Tracking stats:
    losses_i.append(loss.log10().item()) # log10 for better visualisation

    if i % 50 == 0:
        print(f"Epoch: {i} | Loss: {loss.item()}")


losses_i = torch.tensor(losses_i).view(-1, 100).mean(1) 
plt.plot(losses_i)
plt.show()

@torch.no_grad()
def split_loss():
    
    Xte, Yte = generate_batch(batch_size = batch_size, split = "Test")

    # Forward pass
    logits = model(Xte) # (batch_size, 5)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, Yte)

    print(f"TestLoss: {loss.item()}")

split_loss()