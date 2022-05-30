import time 

import torch
#import cv2
import matplotlib.pyplot as plt

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# FLOW

def read_flow(flow_file):
    with open(flow_file, 'rb') as f:
        flow_f = f.read()
    data = np.frombuffer(buffer=flow_f, dtype=np.float32)
    assert(data[0] == 202021.25)
    w, h = np.frombuffer(buffer=flow_f, dtype=np.int32, count=2, offset=4)
    return data[3:].reshape((h, w, 2))


def visualize_image(image, normalize=False, opencv=True):
    image = image[0].permute(1,2,0).cpu().numpy()
    if normalize:
        image = image/255.0
    
    if opencv:
        cv2.imshow('image', image[:, :, [2,1,0]])
        cv2.waitKey(0)
    else:
        plt.imshow(image)
        plt.show()


# TIME

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is  "+ str(time.time() - startTime_for_tictoc)+"  seconds")
    else:
        print("Toc: start time not set")


# Tensorboard
def log_tensorboard(writer, losses, val_loss, psnr, ssim, lpips, lr, timestep, mode='train'):
    for k, v in losses.items():
        writer.add_scalar('Loss/%s/%s' % (mode, k), v.avg, timestep)
    writer.add_scalar('Validation Loss/%s' % mode, val_loss, timestep)    
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if lpips is not None:
        writer.add_scalar('LPIPS/%s' % mode, lpips, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)
