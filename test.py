import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from args import pass_args
from metrics import Metrics, AverageMeter



def set_seed(seed, device):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if seed != 0:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)


def test():
    args, _ = pass_args(_print=True)
    model_name = args.model_name
    device = args.device
    set_seed(args.set_seed, device)

    print("device:", device)

    from model.model import Model

    batch_size = 16
    from data.vimeo90k import get_dataloader
    dataloader = get_dataloader(args.path, mode='test', batch_size=batch_size, num_workers=args.num_workers, shuffle=False)

    image, _ = next(iter(dataloader))
    model = Model(image[0], levels=6, device=device).to(device)
    
    model.load_state_dict(torch.load(model_name+".pt"))
    model.eval() 

    metrics = Metrics()
    
    with torch.no_grad():
        for batch_idx, (frames, image_path) in enumerate(dataloader):
            frame1, frame2, frame3 = frames[0].to(device), frames[1].to(device), frames[2].to(device)
            
            pred, _ = model(frame1, frame3)

            metrics.eval_metrics(pred, frame2)
            
    print("PSNRS:", metrics.psnrs.avg)
    print("SSIM:", metrics.ssims.avg)
    print("SSIM:", metrics.ssims_matlab.avg)
    print("IE", metrics.interpolation_error.avg)
    print("RIFE IE", metrics.ie_rife.avg)


if __name__ == "__main__":
    test()
