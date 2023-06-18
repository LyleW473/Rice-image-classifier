import cv2
import os
import torch

torch.manual_seed(2000)
batch_size = 32
image_folders = os.listdir("Dataset/Images")
print(image_folders)

num_types = len(image_folders)
uniform_t_distribution = torch.ones(num_types, dtype = torch.float64) / num_types # 5 types of rice
uniform_i_distribution = torch.ones(15000, dtype = torch.float64) / 15000 # 15000 images for each type of rice

print(uniform_i_distribution.dtype)
print(uniform_t_distribution.dtype)
print(uniform_i_distribution)
print(uniform_t_distribution)


def img_to_matrix(image_indexes, r_type_indexes):

    # Creates all pixel matrixes in a batch

    # Find all rice types in the selected batch
    rice_types = [image_folders[r_type_i] for r_type_i in r_type_indexes]

    print(rice_types)
    print(image_indexes)
    
    matrices = []

    for img_type, img_num in zip(rice_types, image_indexes):

        # img_num.item() as img_num is a tensor containing the image number
        img_path = f"Dataset/Images/{img_type}/{img_type} ({str(img_num.item())}).jpg" # E.g. Dataset/Images/Jasmine/Jasmine (3).jpg"

        # Read image into numpy array
        img = cv2.imread(img_path)

        # Convert image to grey
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Matrix of shape (62500, 1) with each pixel containing the greyscale value
        pixel_matrix = torch.from_numpy(grey_img).view(250 * 250, 1)
        
        # Add to matrices list
        matrices.append(pixel_matrix)

    # Add all the matrices into a single tensor [Shape: (batch_size, pixel_number, 1)]
    matrices = torch.stack(matrices, dim = 0)
    
    return matrices
    
def generate_batch(batch_size):
    
    # Generate indexes for rice types
    rice_type_indexes = torch.multinomial(input = uniform_t_distribution, num_samples = batch_size, replacement = True)

    # Convert the index into a one-hot vector for the entire batch
    rice_types = torch.zeros(batch_size, 5)
    rice_types[torch.arange(batch_size), rice_type_indexes - 1] += 1 # 2nd option = index 1 (Possible indexes = 0, 1, 2, 3, 4 ... num_types)

    # -------------------------------

    # Generate indexes for rice images
    rice_image_indexes = torch.multinomial(input = uniform_i_distribution, num_samples = batch_size, replacement = True)

    # Convert indexes to matrices
    rice_image_matrices = img_to_matrix(image_indexes = rice_image_indexes, r_type_indexes = rice_type_indexes)


    # Pixel matrices, Labels
    return rice_image_matrices, rice_types
        

Xb, Yb = generate_batch(batch_size = batch_size)

print(Xb.shape, Yb.shape)