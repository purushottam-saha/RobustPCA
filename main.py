from datetime import datetime
import preprocessing
from pcp import pcp
from stoc_rpca import stoc_rpca
from omwrpca import omwrpca
from real_pcp import realpcp
from smoother import smooth


names = ['WavingTrees180-','MovedObject1300-1600','Camouflage','Camouflage2']     # 'MovedObject1300-1600'         #'WavingTrees180-'
algos = ['stocrpca',]
# names = ['WavingTrees180-',]
# algos = ['omwrpca',]
# desired_width = 126 #400    16 to 9 ratio
# desired_height = 84 #225

def do(name,algo):
    if algo=='pcp':
        start_time = datetime.now()
        print("PCP Process started")
        # preprocessing.resize_video(f'Data/{name}.mp4', f'Data/resize_{name}.mp4', desired_width, desired_height)
        # print("Resizing done")
        # preprocessing.to_gif(f'Data/resize_{name}.mp4',f'Data/{name}.gif')
        # print("Gif created")
        M,shape,n_frames = preprocessing.get_X_from_gif(f'Data/{name}.gif')
        #print(M.shape)
        print(f'Data Loaded, {n_frames} frames and {shape[0]}x{shape[1]} size images.')
        preprocessing.mat_to_gif(M,shape,n_frames,f'output/{name}_bnw.gif')
        #print("PCP Running")
        recons, frecons,iteration,rank = pcp(M,maxit=1000)
        print(f'complete, rank = {rank}')
        preprocessing.mat_to_gif(recons,shape,n_frames,f'output/{algo}/{name}_bnw_backg.gif')
        preprocessing.mat_to_gif(frecons,shape,n_frames,f'output/{algo}/{name}_bnw_foreg.gif')
        smooth(f'output/{algo}/{name}_bnw_foreg.gif',f'output/{algo}/{name}_bnw_foreg_smoothed.gif')
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

    elif algo == 'stocrpca':
        start_time = datetime.now()
        print("Stochastic RPCA Process started")
        # preprocessing.resize_video(f'Data/{name}.mp4', f'Data/resize_{name}.mp4', desired_width, desired_height)
        # print("Resizing done")
        # preprocessing.to_gif(f'Data/resize_{name}.mp4',f'Data/{name}.gif')
        # print("Gif created")
        M,shape,n_frames = preprocessing.get_X_from_gif(f'Data/{name}.gif')
        #print(M.shape)
        print(f'Data Loaded, {n_frames} frames and {shape[0]}x{shape[1]} size images.')
        #preprocessing.mat_to_gif(M,shape,n_frames,f'output/{name}_bnw.gif')
        print("Main Process Starting...")
        recons, frecons,rank, U = stoc_rpca(M.transpose(), burnin = 15)
        print(f'Main Process Complete, rank = {rank}')
        preprocessing.mat_to_gif(recons.transpose(),shape,n_frames,f'output/{algo}/{name}_bnw_backg.gif')
        preprocessing.mat_to_gif(frecons.transpose(),shape,n_frames,f'output/{algo}/{name}_bnw_foreg.gif')
        smooth(f'output/{algo}/{name}_bnw_foreg.gif',f'output/{algo}/{name}_bnw_foreg_smoothed.gif')
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

        #smooth(f'output/{algo}/{name}_bnw_foreg.gif',f'output/{algo}/{name}_bnw_foreg_smoothed.gif')
    elif algo == 'omwrpca':
        start_time = datetime.now()
        print("OMWRPCA Process started")
        # preprocessing.resize_video(f'Data/{name}.mp4', f'Data/resize_{name}.mp4', desired_width, desired_height)
        # print("Resizing done")
        # preprocessing.to_gif(f'Data/resize_{name}.mp4',f'Data/{name}.gif')
        # print("Gif created")
        M,shape,n_frames = preprocessing.get_X_from_gif(f'Data/{name}.gif')
        # print(M.shape)
        print(f'Data Loaded, {n_frames} frames and {shape[0]}x{shape[1]} size images.')
        #preprocessing.mat_to_gif(M,shape,n_frames,f'output/{name}_bnw.gif')
        print("Main Process Starting...")
        recons, frecons, rank = omwrpca(M.transpose(), burnin = 15, win_size = 10 )
        print(f'Main Process Complete, rank = {rank}')
        preprocessing.mat_to_gif(recons.transpose(),shape,n_frames,f'output/{algo}/{name}_bnw_backg.gif')
        preprocessing.mat_to_gif(frecons.transpose(),shape,n_frames,f'output/{algo}/{name}_bnw_foreg.gif')
        smooth(f'output/{algo}/{name}_bnw_foreg.gif',f'output/{algo}/{name}_bnw_foreg_smoothed.gif')
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

    # elif algo == 'realpcp':
    #     start_time = datetime.now()
    #     print("Real PCP Process started")
    #     # preprocessing.resize_video(f'Data/{name}.mp4', f'Data/resize_{name}.mp4', desired_width, desired_height)
    #     # print("Resizing done")
    #     # preprocessing.to_gif(f'Data/resize_{name}.mp4',f'Data/{name}.gif')
    #     # print("Gif created")
    #     M,shape,n_frames = preprocessing.get_X_from_gif(f'Data/{name}.gif')
    #     print(M.shape)
    #     print(f'Data Loaded, {n_frames} frames and {shape[0]}x{shape[1]} size images.')
    #     #preprocessing.mat_to_gif(M,shape,n_frames,f'output/{name}_bnw.gif')
    #     #print("PCP Running")
    #     recons, frecons,iteration,rank = realpcp(M,maxit=200)
    #     print(f'complete, rank = {rank}')
    #     preprocessing.mat_to_gif(recons,shape,n_frames,f'output/{algo}/{name}_bnw_backg.gif')
    #     preprocessing.mat_to_gif(frecons,shape,n_frames,f'output/{algo}/{name}_bnw_foreg.gif')
    #     preprocessing.mat_to_gif(smooth(f'output/{algo}/{name}_bnw_foreg.gif'),f'output/{algo}/{name}_bnw_foreg_smoothed.gif')
    #     end_time = datetime.now()
    #     print('Duration: {}'.format(end_time - start_time))



if __name__=='__main__':
    for algo in algos:
        for name in names:
            do(name,algo)