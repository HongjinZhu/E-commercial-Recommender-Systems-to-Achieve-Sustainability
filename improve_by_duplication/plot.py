import matplotlib.pyplot as plt
import numpy as np

x = [0.594, 0.614,0.630,0.645,0.661,0.671,0.683,0.683,0.692,0.691,0.694,0.698,0.698,0.701,0.693,0.691,0.701,0.699,0.696,0.701]
print(len(x))
y = [0.055,0.050,0.050,0.050,0.048,0.044,0.042,0.039,0.037,0.036,0.036,0.035,0.034,0.035,0.033,0.034,0.034,0.033,0.034,0.034]
print(len(y))
z = [0.366,0.383,0.395,0.412,0.432,0.446,0.466,0.474,0.488,0.490,0.498,0.504,0.501,0.508,0.504,0.501,0.510,0.509,0.506,0.515]
print(len(z))

# plt.scatter(x,y)
# plt.xlabel('HR')
# plt.ylabel('Sustainability Proportion')
# plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(x, y)
# ax1.set_title('Sharing Y axis')
ax2.scatter(z, y)
ax1.set_xlabel('HR')
ax1.set_ylabel('Sustainability Proportion')
ax2.set_xlabel('NDCG')
plt.show()
