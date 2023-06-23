import cv2
import os
import torch
from torch import nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(2000)

batch_size = 32
image_folders = os.listdir("Dataset/Images")
device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Device: {device}")
image_size = (100, 100)
num_image_pixels = image_size[0] * image_size[1]

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

    # Holds the losses for the train split and test split (with no change in model parameters)
    split_losses = {}

    for split in ("Train", "Test"):

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

            # Test accuracy
            if split == "Test":
                # Find the accuracy on the predictions on this batch
                accuracies[x] = (count_correct_preds(predictions = logits, targets = Yev).item() / batch_size) * 100 # Returns tensor containing the number of correct predictions
                # print(f"Accuracy on batch: {accuracies[x]}")

        split_losses[split] = losses.mean()
        avg_test_accuracy = accuracies.mean()
    
    model.train() 

    return split_losses, avg_test_accuracy

def count_correct_preds(predictions, targets):
    # Find the predictions of the model
    _, output = torch.max(predictions, dim = 1) 
    output = F.one_hot(output, num_classes = 5) # 5 types of rice

    # Return the number of correct predictions
    return torch.sum((output == targets).all(axis = 1))
        
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

                    # 2 
                    nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1), 
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2),

                    nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2),

                    nn.Flatten(),

                    nn.Linear(32 * 25 * 25, 128),
                    nn.ReLU(),
                    nn.Linear(128, 5) # 5 types of rice

                    )
                    
model.to(device = device)

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
optimiser = torch.optim.AdamW(model.parameters(), lr = 1e-3) # Adam (updates learning rate for each weight individually)

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
        split_losses, test_acc = evaluate_loss(num_iterations = 20)
        print(f"Epoch: {i} | TrainLoss: {split_losses['Train']:.4f} | TestLoss: {split_losses['Test']:.4f} | AverageTestAccuracy: {test_acc}%")
        accuracies.append(test_acc)


losses_i = torch.tensor(losses_i).view(-1, 100).mean(1) 
plt.plot(losses_i)
plt.show()

# Evaluate model
model.eval()
split_loss("Train")
split_loss("Test")
print(f"AvgTestAccuracy: {sum(accuracies) / len(accuracies)}") # Average test accuracy overall when the model was training

test_losses_i = []
num_correct = 0
test_steps = 500
test_batch_size = 30
# 15000 total test images if test_split_multiplier == 0.2
with torch.no_grad():
    
    for i in range(test_steps): # Test 500 * 30 images in the test split
        Xte, Yte = generate_batch(batch_size = test_batch_size, split = "Test")

        logits = model(Xte)
        loss = F.cross_entropy(logits, Yte)

        num_correct += count_correct_preds(predictions = logits, targets = Yte)
        test_losses_i.append(loss.log10().item())

        if i % 50 == 0 and i != 0:
            print(f"Correct predictions: {num_correct} / {i * test_batch_size} | Accuracy(%): {(num_correct / (i * test_batch_size)) * 100}")

test_losses_i = torch.tensor(test_losses_i).view(-1, 100).mean(1) 
plt.plot(test_losses_i)
plt.show()
print(f"Correct predictions: {num_correct} / {test_steps * test_batch_size} | Accuracy(%): {(num_correct / (test_steps * test_batch_size)) * 100}")