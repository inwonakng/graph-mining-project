import matplotlib.pyplot as plt
import numpy as np
# right now this is just for dataset 1
acc_types = ['Total','White victory','Black victory','Draw']
test_types = ['Common Neighbors','Common Neighbors (Colors)','Edge Weights','Edge Weights (Colors)',
            'Number Paths (3)', 'Number Paths (4)','Opening Moves', 'Pageranks (100)', 'Pageranks (1000)']
common_neighbors_acc = [55.31,	57.55,	55.49,	3.47]
ccc_acc = [55.25,	56.63,	57.06,	3.45]
edge_weights_acc = [58.22,	59.52,	57.04,	4.65]
ewc_acc = [57.69,	57.87,	57.78,	2]
num_paths_3_acc = [44.9,	56.4,	54.4,	5.16]
num_paths_4_acc = [55.26,	57.52,	55.4,	3.53]
om_acc = [43.56,	55.97,	53.65,	3.32]

a = [13.13,	51.7,	48.55,	3.67]
b = [13.58,	52.25,	50.12,	3.72]
c = [13.29,	51.78,	50.18,	3.61]

pageranks_100 = [(i+j+k)/3 for i,j,k in zip(a,b,c)]

aa = [36.75,	52.49,	50.08,	3.4]
bb = [36.8,	52.48,	50.21,	3.43]
cc = [36.04,	52.34,	49.96,	3.58]

pageranks_1000 = [(i+j+k)/3 for i,j,k in zip(aa,bb,cc)]

data = [common_neighbors_acc,ccc_acc, edge_weights_acc,ewc_acc ,num_paths_3_acc,num_paths_4_acc,om_acc,pageranks_100,pageranks_1000]
X = np.arange(4)
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(111)

ax.set_title('Accuracy values for different methods (Dataset 3)')
ax.set_ylabel('Percentage')

l0 = ax.bar(X+0.00,data[0],color = 'c',width = 0.10)

l1 = ax.bar(X+0.10,data[1],color = '#6e2aeb',width = 0.10)

l2 = ax.bar(X+0.20,data[2],color = '#ebb12a',width = 0.10)

l3 = ax.bar(X+0.30,data[3],color = '#eb2adb',width = 0.10)

l4 = ax.bar(X+0.40,data[4],color = '#e3e032',width = 0.10)

l5 = ax.bar(X+0.50,data[5],color = '#2dc25c',width = 0.10)

l6 = ax.bar(X+0.60,data[6],color = 'r',width = 0.10)

l7 = ax.bar(X+0.70,data[7],color = '#42b6f5',width = 0.10)

l8 = ax.bar(X+0.80,data[8],color = '#d7f542',width = 0.10)

ax.legend((l0,l1,l2,l3,l4,l5,l6,l7,l8),(test_types))

ax.set_xticks( np.arange(0,4)+.375)
ax.set_xticklabels(acc_types)
ax.set_yticks(np.arange(0,101,10))

# plt.show()
plt.savefig('imgs/dataset3.png')
