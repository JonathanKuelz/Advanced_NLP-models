import argparse
import torch


def parse():
    """
    Reads information from terminal input and processes them.
    Builds up on: https://docs.python.org/3/library/argparse.html#action
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='entnet', help='entnet or lstm')
    parser.add_argument("--task", type=str, default='babi')
    parser.add_argument("--dataset_path", type=str, default='/storage/remote/atcremers45/s0238/en-10k/')
    parser.add_argument("--babi_task", type=int, default=1, help='bAbI task to train and validate on')
    parser.add_argument("--load_model", type=str, default=None, help='Path to saved classifier model')
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--njobs", type=int, default=4, help='Number of Workers for Torch dataloader')
    parser.add_argument("--lr", type=float, default=0.02, help='Learning Rate')
    parser.add_argument("--num_blocks", type=int, default=20, help='Blocks in the EntNet')
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # TODO: Infinite duration of length only interrupted by sigterm signal
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs after which training stops')
    parser.add_argument("--save_interval", type=int, default=10, help='Save model every n-th epoch')
    parser.add_argument("--output_path", type=str, default='/usr/prakt/s0238/entnetmodels',
                                help='Location to save the logs')
    parser.add_argument("--name", type=str, default='default_model', help='Experiment Name')
    parser.add_argument("--gpuid", type=int, default=0, help='Default GPU id')  # TODO: What is this
    parser.add_argument("--multi", action='store_true', help='To use DataParallel')  # TODO: How exactly is it working?
    parser.add_argument("--gpu_range", type=str, default="0,1,2,3", help='GPU ids to use if multi')  # TODO: See above
    parser.add_argument("--cyc_lr", action='store_true', help='Cyclic LR')
    parser.add_argument("--cyc_step_size_up", type=int, default=2000, help='Steps the cyclic learning rate goes up before resetting')
    parser.add_argument("--optimizer", type=str, default="adam", help="Choose adam or sgd")

    args = parser.parse_args()
    args.gpu_range = [int(_) for _ in args.gpu_range.split(",")]
    args.device = torch.device("cuda:%d"%args.gpuid if torch.cuda.is_available() else "cpu")  # TODO: Reformat string
    return args
