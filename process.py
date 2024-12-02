from PIL import Image
import numpy as np
import os
import preprocessing

# For the processing of Wallflower datasets

def bmp_to_vector(file_path):
    # Open the image file
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)  # Convert to a NumPy array
    return img_array.flatten(),np.shape(img_array)  # Flatten to a 1D vector

def process_directory(directory,exceptt = None):
    """
    Process all BMP files in a directory and convert to vectors.
    Returns:
        dict: A dictionary with filenames as keys and vectors as values.
    """
    vectors = []
    for filename in os.listdir(directory):
        #print(filename[2:6])
        if exceptt:
            if filename in exceptt:
                continue
        if filename.lower().endswith('.bmp'):
            if (int(filename[2:6])>1600) or (int(filename[2:6])<1300):
                continue
            file_path = os.path.join(directory, filename)
            tmp = bmp_to_vector(file_path)
            vectors.append(tmp[0])
    vectors = np.array(vectors)
    return vectors,tmp[1]

# Example usage
bmp_directory = 'Data/MovedObject/'
vectors,shape = process_directory(bmp_directory,exceptt=['hand_segmented_00985.BMP'])
preprocessing.mat_to_gif(vectors,shape,len(vectors),f'{bmp_directory[:-1]}.gif')

# # Print the vector of the first file
# for filename, vector in vectors.items():
#     print(f"{filename}: {vector[:10]}...")  # Print the first 10 values for brevity
#     break
