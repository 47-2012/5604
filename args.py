import argparse

parser = argparse.ArgumentParser(description='Parameter arguments')

# Dataset
parser.add_argument('--dataset', type=str, default='vimeo90k', choices=['vimeo90k, snufilm (not supported)', 'ucf101', 'middlebury'])
parser.add_argument('--path', type=str, default='data/vimeo90k_triplet')

# Training
parser.add_argument('--mode', type=str, default='training')
parser.add_argument('--loss', type=str, default='L1')
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--set_seed', type=int, default=0)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--log_iter', type=int, default=1000)
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--test', action='store_true', default=True)
parser.add_argument('--exp_name', type=str, default='test')


# Testing
parser.add_argument('--model_name', type=str, default='iter100_lap_e5')
parser.add_argument('--save_path', type=str, default='data/predictions/')

# CPU/GPU
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_workers', type=int, default=3)


def pass_args(_print=True):
    args, unknownargs = parser.parse_known_args()
    if _print:
        print(args)
    return args, unknownargs
