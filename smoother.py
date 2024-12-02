from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np
from sklearn.decomposition import PCA
from moviepy.editor import VideoFileClip
import PIL 
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import cv2

def get_X_from_gif_smoothen(filename, kernel_size=3,thresh=170):
    gif = Image.open(filename)
    gif.seek(0)
    images = []
    shape = None
    first = True
    try:
        while True:
            # Convert GIF frame to a non-palette image (RGB)
            tmp = gif.convert("L")
            a = np.asarray(tmp)
            
            if first:
                shape = a.shape[:2]  # Store the height and width of the image
                first = False

            # Smooth the frame using neighboring pixels
            smoothed_frame = smooth_frame(a, kernel_size).flatten()
            images.append([0 if ff<thresh else ff for ff in smoothed_frame])
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    return np.asarray(images), shape, gif.n_frames

def smooth_frame(frame, kernel_size=3):
    """
    Smooth a frame using neighboring pixels with a kernel.
    """
    height, width = frame.shape

    # Pad the frame to handle edge cases
    padded_frame = cv2.copyMakeBorder(frame, kernel_size // 2, kernel_size // 2,
                                      kernel_size // 2, kernel_size // 2, cv2.BORDER_REFLECT)

    # Initialize the smoothed frame
    smoothed_frame = np.zeros_like(frame)

    # Smooth each pixel in the frame
    for i in range(height):
        for j in range(width):
            # Extract the neighborhood
            neighborhood = padded_frame[i:i + kernel_size, j:j + kernel_size]
            # Compute the mean of the neighborhood
            mean_value = np.mean(neighborhood)
            # Set the smoothed value
            smoothed_frame[i, j] = int(mean_value)

    return smoothed_frame

def mat_to_gif(X,shape,n_frames,filename,fps=30):
    #array = np.array([[[0 for i in range(shape[1])] for j in range(shape[0])] for k in range(n_frames)])
    #for k in range(n_frames):
    #    for j in range(shape[0]):
    #        for i in range(shape[1]):
    #            array[k,j,i] = X[k][i+j*shape[1]]
    array = X.reshape([n_frames,*shape])
    array = array[..., np.newaxis]  * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps)
    clip.write_gif(filename, fps=fps)
    return clip


def smooth(filename,outfile, kernel_size = 2):
    X,shape,n_frames = get_X_from_gif_smoothen(filename, kernel_size=2)
    mat_to_gif(X,shape,n_frames,outfile)

if __name__=='__main__':
    print("Process started")
    X,shape,n_frames = get_X_from_gif_smoothen("new/tom and jerry_bnw_foreg.gif", kernel_size=2)
    print(f'Data Loaded, {n_frames} frames and {shape[0]}x{shape[1]} size images.')
    mat_to_gif(X,shape,n_frames,'new/tomnjerry_back.gif')