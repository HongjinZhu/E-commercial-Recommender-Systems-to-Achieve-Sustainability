# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = ''

# train_rating = main_path + '{}.train.rating.csv'.format(dataset)
train_rating = 'ml-1m.train.rating.csv'
test_rating = main_path + '{}.test.rating.csv'.format(dataset)
test_negative = main_path + '{}.test.negative.csv'.format(dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
