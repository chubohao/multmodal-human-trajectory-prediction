import argparse
import torch

"""
Tiny utility to print the command-line args used for a checkpoint
"""

parser = argparse.ArgumentParser()
#parser.add_argument('--checkpoint', default='/home/martin-pc/PycharmProjects/crowd-sim/models/sgan-p-models/zara2_8_model.pt')
#parser.add_argument('--model_path',default='/home/martin/sgan/models/sgan-p-models/hotel_8_model.pt', type=str)
parser.add_argument('--checkpoint',default='/home/martin-pc/Documents/col_aware/SR-LSTM-univ--new5coll_loss_balance_a001/checkpoint_with_model.pt', type=str)

def main(args):
	checkpoint = torch.load(args.checkpoint, map_location='cpu')
	for k, v in checkpoint['args'].items():
		print(k, v)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)