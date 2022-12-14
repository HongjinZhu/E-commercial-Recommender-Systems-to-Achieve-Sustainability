import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils
# import faulthandler
# faulthandler.enable()


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=1024,
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat, sust = data_utils.load_all()
# print('train_data', train_data)
# print('test_data',test_data)

# construct the train and test datasets
train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True,sust)
print('qbc', train_dataset)
test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False,None)
print('qdbc', test_dataset)

train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=4)
# for user, item, label,sust in train_loader:
# 	print(222,sust)s
test_loader = data.DataLoader(test_dataset,batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
# model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
if __name__ == "__main__":
	count, best_hr = 0, 0

	for epoch in range(args.epochs):
		model.train() # Enable dropout (if have).
		start_time = time.time()
		train_loader.dataset.ng_sample()
		freq_0 = 0.98
		freq_1 = 0.02
		theta = 0.9

		for user, item, label,t_sust in train_loader:
			# print(222,sust)
			# user = user.cuda()
			# item = item.cuda()
			# label = label.float().cuda()

			# if is_sus == 1:
			u1 = user[(t_sust != 0).nonzero()].reshape(-1)
			t1 = item[(t_sust != 0).nonzero()].reshape(-1)
			l1 = label[(t_sust != 0).nonzero()].reshape(-1)

			l1 = l1.float()
			model.zero_grad()
			prediction1 = model(u1, t1)
			loss1 = loss_function(prediction1, l1)

			# elif is_sus == 0:
			u0 = user[(t_sust == 0).nonzero()].reshape(-1)
			t0 = item[(t_sust == 0).nonzero()].reshape(-1)
			l0 = label[(t_sust == 0).nonzero()].reshape(-1)

			l0 = l0.float()
			model.zero_grad()
			prediction0 = model(u0, t0)
			loss0 = loss_function(prediction0, l0)

			# prediction = model(user, item)

			loss = 0.999*loss0 + 0.001*loss1
			# loss = loss_function(prediction, label)
			loss.backward()
			optimizer.step()
			# writer.add_scalar('data/loss', loss.item(), count)
			count += 1

		model.eval()
		HR, NDCG, Sustainable_Prop = evaluate.metrics(model, test_loader, args.top_k, sust)

		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
				time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print("HR: {:.3f}\tNDCG: {:.3f}\tProportion of Sustainable in Recommends: {:.3f}".format(np.mean(HR), np.mean(NDCG), np.mean(Sustainable_Prop)))

		if HR > best_hr:
			best_hr, best_ndcg, best_epoch, bestSus = HR, NDCG, epoch, Sustainable_Prop
			if args.out:
				if not os.path.exists(config.model_path):
					os.mkdir(config.model_path)
				torch.save(model,
					'{}{}.pth'.format(config.model_path, config.model))

	print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}, Proportion of Sustainable in Recommends: {:.3f}".format(
										best_epoch, best_hr, best_ndcg, bestSus))
