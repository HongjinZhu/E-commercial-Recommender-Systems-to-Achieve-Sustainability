import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0

def hit_sust(sust, pred_items):
	num_hit = len(set(sust).intersection(set(pred_items)))
	# print(num_hit)
	return num_hit/10


def metrics(model, test_loader, top_k, sust):
	HR, NDCG, Sustainable_Prop = [], [], []

	for user, item, label, t_sust in test_loader:
		# user = user.cuda()
		# item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()
		# print(recommends)

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))
		Sustainable_Prop.append(hit_sust(sust,recommends))

	return np.mean(HR), np.mean(NDCG), np.mean(Sustainable_Prop)
