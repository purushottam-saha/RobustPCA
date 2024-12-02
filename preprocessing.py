from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np
from sklearn.decomposition import PCA
from moviepy.editor import VideoFileClip
import PIL 
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

def resize_video(input_path, output_path, width, height):
    # Load the video clip
    video = VideoFileClip(input_path)

    # Resize the video clip
    resized_video = video.resize((width, height))

    # Write the resized video to the output file
    resized_video.write_videofile(output_path, fps=30, codec='libx264', audio_codec='aac')
    
    # Close the video clip
    video.close()

def get_X_from_gif(filename):
    gif = Image.open(filename)
    gif.seek(0)
    images = []
    shape = None
    first = True
    try:
        while True:
            # Get image as numpy array
            tmp = gif.convert('L') # Make without palette
            a = np.asarray(tmp)
            if first:
                shape=a.shape[:2]
            if len(a.shape)==0:
                raise MemoryError("Too little memory to convert PIL image to array")
            #print(a.shape)
            b = np.array([0 for _ in range(shape[0]*shape[1])])
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    b[i*shape[1]+j] = a[i,j]
            # Store, and next
            images.append(b)
            gif.seek(gif.tell()+1)
    except EOFError:
        pass
    return np.asarray(images),shape,gif.n_frames

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

def do_pca(X,shape,n_frames,comps,filename):
    pca = PCA(n_components=comps)
    fitted = pca.fit_transform(X)
    recons = pca.inverse_transform(fitted)
    print("PCA performed")
    mat_to_gif(recons,shape,n_frames,f'{filename}_backg.gif')
    mat_to_gif(X-recons,shape,n_frames,f'{filename}_foreg.gif')

def to_gif(input_path, output_path):
    videoClip = VideoFileClip(input_path)
    videoClip.write_gif(output_path)


# Example usage
#input_file = 'mp4vid/fan hand.mp4'
#output_file = 'mp4vid/resizefan hand.mp4'
#desired_width = 225
#desired_height = 400
#resize_video(input_file, output_file, desired_width, desired_height)


if __name__=='__main__':
    print("Process started")
    X,shape,n_frames = get_X_from_gif('room hand.gif')
    print(f'Data Loaded, {n_frames} frames and {shape[0]}x{shape[1]} size images.')
    mat_to_gif(X,shape,n_frames,'room hand_bnw.gif')
    #mat_to_gif(X,shape,n_frames,'recreated.gif')
    do_pca(X,shape,n_frames,2,'room hand')