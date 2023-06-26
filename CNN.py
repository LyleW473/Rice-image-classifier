import cv2
import os
import torch
from torch import nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(2000)

batch_size = 20
image_folders = os.listdir("Dataset/Images")
device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Device: {device}")
image_size = (100, 100)
num_image_pixels = image_size[0] * image_size[1]

# Splits and distributions for generating batches
num_types = len(image_folders)
uniform_types_distribution = torch.ones(num_types, dtype = torch.float32, device = device) / num_types # 5 types of rice

# 5 types of rice, 15000 images for each = 75000 total
# 75000 * 0.1 = 7500 images in total for test/validation split
# 7500 / 5 = 1500 images of each type for test/validation split
test_split_multiplier = 0.1
val_split_multiplier = 0.1

num_test_imgs = int((75000 * test_split_multiplier) / 5)
num_val_imgs = int((75000 * val_split_multiplier) / 5)
num_train_imgs = int(15000 - num_test_imgs - num_val_imgs)
print(num_train_imgs, num_val_imgs, num_test_imgs)

uniform_train_images_distribution = torch.ones(num_train_imgs, dtype = torch.float32, device = device) / num_train_imgs # 15000 images for each type of rice
uniform_test_images_distribution = torch.ones(num_test_imgs, dtype = torch.float32, device = device) / num_test_imgs
uniform_val_images_distribution = torch.ones(num_val_imgs, dtype = torch.float32, device = device) / num_val_imgs

# print(uniform_train_images_distribution.dtype)
# print(uniform_types_distribution.dtype)
# print(uniform_types_distribution)
# print(uniform_train_images_distribution)

def img_to_matrix(image_indexes, r_type_indexes, img_size):

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
        img_np = cv2.imread(img_path)

        # Scale down image
        img_np = cv2.resize(img_np, img_size)

        # Create PyTorch tensor from numpy array
        img_pt = torch.from_numpy(img_np).float()

        # Add to matrices list
        matrices.append(img_pt.view(3, image_size[0], image_size[1]))

        # Release image memory
        del img_np, img_pt
        

    # Add all the matrices into a single tensor [Shape = (batch_size, 3(R, G, B channels), image_size[0], image_size[1]]
    matrices = torch.stack(matrices, dim = 0)
    
    return matrices
    
def generate_batch(batch_size, split):
    
    # Generate indexes for rice types
    rice_type_indexes = torch.multinomial(input = uniform_types_distribution, num_samples = batch_size, replacement = True)

    # Convert the index into a one-hot vector for the entire batch
    rice_types = torch.zeros(batch_size, 5, device = device)
    rice_types[torch.arange(batch_size), rice_type_indexes - 1] += 1 # 2nd option = index 1 (Possible indexes = 0, 1, 2, 3, 4 ... num_types)
    
    # -------------------------------

    # Train split images
    if split == "Train":
        # Generate indexes for rice images
        rice_image_indexes = torch.multinomial(input = uniform_train_images_distribution, num_samples = batch_size, replacement = True)
        rice_image_indexes += 1 # The indexes only go from 0 - 14999 but the numbers at the end of each image go from 1 - 15000
    
    # Val split images
    elif split == "Val":
        # Generate indexes for rice images
        rice_image_indexes = torch.multinomial(input = uniform_val_images_distribution, num_samples = batch_size, replacement = True)
        # Note: If val_split_multiplier = 0.1: 15000 - 1500 - 1500 + 1 = 12001 
        # i.e. images at indexes 12001 - 13500 for each rice type are for the val split
        rice_image_indexes += (15000 - num_test_imgs - num_val_imgs) + 1 # The indexes only go from 0 - 14999 but the numbers at the end of each image go from 1 - 15000
    
    # Test split images
    else:
        # Note: uniform_test_images_distribution will have a smaller range of index values 
        # - (e.g. if num_test_imgs = 7500, as there are 5 types, the last (7500 / 5) = 1500 images of this type are for the test split)
        # - i.e. Only images at indexes 13501 - 15000 will be used for the test split
        rice_image_indexes = torch.multinomial(input = uniform_test_images_distribution, num_samples = batch_size, replacement = True)
        rice_image_indexes += (15000 - num_test_imgs) + 1 # The indexes only go from 0 - 14999 but the numbers at the end of each image go from 1 - 15000

    # Convert indexes to matrices
    rice_image_matrices = img_to_matrix(image_indexes = rice_image_indexes, r_type_indexes = rice_type_indexes, img_size = image_size)

    # Pixel matrices, Labels
    return rice_image_matrices.to(device = device), rice_types.to(device = device)
        
@torch.no_grad()
def split_loss(split):
    
    X, Y = generate_batch(batch_size = batch_size, split = split)

    # Forward pass
    logits = model(X) # (batch_size, 5)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, Y)

    print(f"{split}Loss: {loss.item()}")

@torch.no_grad()
def evaluate_loss(num_iterations):
    
    model.eval()

    # Holds the losses for the train split and val split (with no change in model parameters)
    split_losses = {}

    for split in ("Train", "Val"):

        losses = torch.zeros(num_iterations, device = device)
        accuracies = torch.zeros(num_iterations, device = device)

        for x in range(num_iterations):
            Xev, Yev = generate_batch(batch_size = batch_size, split = split)

            # Forward pass
            logits = model(Xev)
            # Cross-Entropy loss
            loss = F.cross_entropy(logits, Yev)
            # Set loss
            losses[x] = loss.item()

            # Val accuracy
            if split == "Val":
                # Find the accuracy on the predictions on this batch
                accuracies[x] = (count_correct_preds(predictions = logits, targets = Yev) / batch_size) * 100 # Returns tensor containing the number of correct predictions
                # print(f"Accuracy on batch: {accuracies[x]}")

        split_losses[split] = losses.mean()
        avg_val_accuracy = accuracies.mean()
    
    model.train() 

    return split_losses, avg_val_accuracy

def count_correct_preds(predictions, targets):
    # Find the predictions of the model
    _, output = torch.max(predictions, dim = 1) 
    output = F.one_hot(output, num_classes = 5) # 5 types of rice

    # Return the number of correct predictions
    return torch.sum((output == targets).all(axis = 1)).item()
        
# No.of inputs = Number of pixels in image 
model = nn.Sequential(

                    # Input shape = (batch_size, 1, image_size[0], image_size[1])

                    # For convulational layers, padding = 0, stride = 1, dilation = 1

                    # In PyTorch the output shape after a convulational layer and a maxpool layer can be calculated with the formula
                    # output_size[0] = [(input_size[0] + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]] + 1
                    # output_size[1] = [(input_size[1] + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]] + 1

                    # Conv1
                    # output_size = [(100 + (2 * 0) - (1 * (3 - 1)) - 1) / 1] + 1 ---> 98
                    # (20, 1, 100, 100) ---> (20, 32 [out_features], 98, 98)

                    # MaxPool1 (stride == kernel_size)
                    # output_size = [(98 + (2 * 0) - (1 * (2 - 1)) - 1) / 2] + 1 ---> 49
                    # (20, 32, 98, 98) ---> (20, 32 [out_features], 49, 49)

                    # Conv2
                    # output_size = [(49 + (2 * 0) - (1 * (3 - 1)) - 1) / 1] + 1 ---> 47
                    # (20, 32, 49, 49) ---> (20, 64 [out_features], 47, 47)

                    # MaxPool2 (stride == kernel_size)
                    # output_size = [(47 + (2 * 0) - (1 * (2 - 1)) - 1) / 2] + 1 ---> 23.5 ---> 23
                    # (20, 64, 47, 47) ---> (20, 64, 23, 23)

                    # Conv3 
                    # output_size = [(23 + (2 * 0) - (1 * (3 - 1)) - 1) / 1] + 1 ---> 21
                    # (20, 64, 23, 23) ---> (20, 64 [out_features], 21, 21)     


                    # Flatten
                    # (20, 64, 21, 21) ---> (20, 64 * 21 * 21) ---> (20, 28224)


                    # # 1
                    # nn.Conv2d(1, 32, kernel_size = 3), 
                    # nn.ReLU(),
                    # nn.MaxPool2d(kernel_size = 2),

                    # nn.Conv2d(32, 64, kernel_size = 3),
                    # nn.ReLU(),
                    # nn.MaxPool2d(kernel_size = 2),

                    # nn.Conv2d(64, 64, kernel_size = 3),
                    # nn.ReLU(),
                    
                    # nn.Flatten(), # Convert to 1-D tensor

                    # nn.Linear(28224, 128),
                    # nn.ReLU(),
                    # nn.Linear(128, 5) # 5 types of rice

                    # # 2 
                    # nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1), 
                    # nn.ReLU(),
                    # nn.MaxPool2d(kernel_size = 2, stride = 2),

                    # nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
                    # nn.ReLU(),
                    # nn.MaxPool2d(kernel_size = 2, stride = 2),

                    # nn.Flatten(),

                    # # Note: To find out the in_features, comment out everything after the nn.Flatten() and find the shape of the output of nn.MaxPool2d
                    # # The last 3 numbers of the shape should be the in_features for the first linear layer
                    # nn.Linear(32 * 25 * 25, 128), 
                    # nn.ReLU(),
                    # nn.Linear(128, 5) # 5 types of rice

                    # 3
                    nn.Conv2d(3, 30, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(30), # BatchNorm2d applies batch-norm to a 4-D input (batch_size, channels, height, width)
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2),

                    nn.Conv2d(30, 60, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(60),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2),
 
                    nn.Flatten(),

                    nn.Linear(60 * 25 * 25, 7500),
                    nn.BatchNorm1d(7500),
                    nn.ReLU(),

                    nn.Linear(7500, 1500),
                    nn.BatchNorm1d(1500),
                    nn.ReLU(),
                    
                    nn.Dropout1d(p = 0.1, inplace = False), # "inplace = False" because "inplace = True" modifies the inputs without making a copy tensor (but we need the original when calling loss.backward())
                    nn.Linear(1500, 5)

                    )

model.to(device = device)

# Initialisation

with torch.no_grad():

    # Make the last layer less confident at initialisation
    model[-1].weight *= 0.1

    # Kai-ming initialisation
    for layer in model:
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, mode = "fan_in", nonlinearity = "relu")
            # print(layer.weight.std(), layer.weight.mean())

            # 2nd method:
            # fan_in = layer.weight.size(1)
            # std = torch.sqrt(torch.tensor(2.0 / fan_in))
            # nn.init.normal(layer.weight, mean = 0, std = std)
            # print(layer.weight.std(), layer.weight.mean())

# Optimisers
# optimiser = torch.optim.SGD(model.parameters(), lr = 0.1) # Stochastic gradient descent
optimiser = torch.optim.AdamW(model.parameters(), lr = 0.0005) # Adam (updates learning rate for each weight individually)

Xtr, Ytr = generate_batch(batch_size = batch_size, split = "Train")
print(Xtr.shape)

losses_i = []
accuracies = []

for i in range(20000):
    
    # Generate batch of images
    Xtr, Ytr = generate_batch(batch_size = batch_size, split = "Train")

    # Forward pass
    logits = model(Xtr)

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
        split_losses, val_acc = evaluate_loss(num_iterations = 20)
        print(f"Epoch: {i} | TrainLoss: {split_losses['Train']:.4f} | ValLoss: {split_losses['Val']:.4f} | AverageValAccuracy: {val_acc}%")
        accuracies.append(val_acc)


losses_i = torch.tensor(losses_i).view(-1, 100).mean(1) 
plt.plot(losses_i)
plt.show()

# Evaluate model
model.eval()
split_loss("Train")
split_loss("Val")
split_loss("Test")
print(f"AvgValAccuracy: {sum(accuracies) / len(accuracies)}") # Average validation accuracy overall whilst the model was training

test_losses_i = []
num_correct = 0
num_tested = 0
test_steps = 500
test_batch_size = 30
with torch.no_grad():
    
    for i in range(test_steps):
        Xte, Yte = generate_batch(batch_size = test_batch_size, split = "Test")

        logits = model(Xte)
        loss = F.cross_entropy(logits, Yte)

        num_correct += count_correct_preds(predictions = logits, targets = Yte)
        num_tested += test_batch_size
        test_losses_i.append(loss.log10().item())
        
        if (i + 1) % 50 == 0: # i = 99, this is the 100th iteration
            print(f"Correct predictions: {num_correct} / {num_tested} | Accuracy(%): {(num_correct / num_tested) * 100}")

test_losses_i = torch.tensor(test_losses_i).view(-1, 100).mean(1) 
plt.plot(test_losses_i)
plt.show()


# Set-up 2: (Updated learning rate from 1e-3 to 5e-4 as the model was learning too fast)

# ----------------------------------
# (20 batch-size)
# 20000 steps + Kai-Ming initialised

# TrainLoss: 0.000189362617675215
# ValLoss: 0.9503949284553528
# TestLoss: 1.014149785041809
# AvgValAccuracy: 88.78062438964844
# Correct predictions: 13543 / 15000 | Accuracy(%): 90.28666666666668

# ----------------------------------
# (32 batch-size)
# 20000 steps + Kai-Ming initialised

# TrainLoss: 2.039352875726763e-05
# ValLoss: 1.143798589706421
# TestLoss: 0.02179698273539543
# AvgValAccuracy: 89.6832046508789
# Correct predictions: 13813 / 15000 | Accuracy(%): 92.08666666666666

# ----------------------------------
# (50 batch-size)
# 20000 steps + Kai-Ming initialised

# TrainLoss: 0.005304774735122919
# ValLoss: 1.1827752590179443
# TestLoss: 0.3905371427536011
# AvgValAccuracy: 89.80329132080078
# Correct predictions: 13630 / 15000 | Accuracy(%): 90.86666666666666


# --------------------------------------------------------------------
# Set-up 3:

# ----------------------------------
# (20 batch-size)

# 20000 steps + Kai-Ming initialised
# TrainLoss: 0.0001246255123987794
# ValLoss: 0.0008903587586246431
# TestLoss: 0.0010815162677317858
# AvgValAccuracy: 96.1512451171875
# Correct predictions: 14922 / 15000 | Accuracy(%): 99.48

# 20000 steps + Kai-Ming initialised + dropout(p = 0.2)
# TrainLoss: 0.008496998809278011
# ValLoss: 0.003705524606630206
# TestLoss: 0.04365408793091774
# AvgValAccuracy: 94.66624450683594
# Correct predictions: 14843 / 15000 | Accuracy(%): 98.95333333333333

# 20000 steps + Kai-Ming initialised + dropout(p = 0.1)
# TrainLoss: 0.0004942810628563166
# ValLoss: 0.00016148066788446158
# TestLoss: 0.00033487920882180333
# AvgValAccuracy: 95.13812255859375
# Correct predictions: 14869 / 15000 | Accuracy(%): 99.12666666666667

# 20000 steps + Kai-Ming initialised + dropout(p = 0.1) + less confident linear layer at initialisation
# TrainLoss: 0.002749544335529208
# ValLoss: 9.928843792295083e-05
# TestLoss: 0.0004932366427965462
# AvgValAccuracy: 95.1624984741211
# Correct predictions: 14934 / 15000 | Accuracy(%): 99.56

# ----------------------------------
# (32 batch-size)
# 20000 steps + Kai-Ming initialised

# TrainLoss: 8.250022801803425e-05
# ValLoss: 4.983106555300765e-05
# TestLoss: 0.000184959004400298
# AvgValAccuracy: 97.0484390258789
# Correct predictions: 14921 / 15000 | Accuracy(%): 99.47333333333333

# ----------------------------------
# (50 batch-size)
# 20000 steps + Kai-Ming initialised

# TrainLoss: 6.487225164164556e-06
# ValLoss: 8.530884952051565e-05
# TestLoss: 4.414386785356328e-05
# AvgValAccuracy: 97.43045806884766
# Correct predictions: 14849 / 15000 | Accuracy(%): 98.99333333333334