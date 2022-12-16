import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

l1 = [0, 0, 0,1,0, 0,1, 0,1,1,0]
t1 = torch.tensor(l1)
# print(t1)
print(t1[(t1 != 0).nonzero()].reshape(-1))
print(t1[(t1 == 0).nonzero()])
# print(t1[torch.nonzero(t1)])
# print(torch.is_zero(t1))
