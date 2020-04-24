import matplotlib.pyplot as plt
import numpy as np
# right now this is just for dataset 1
acc_types = ['Total','White victory','Black victory','Draw']
test_types = ['Common Neighbors','Edge Weights', 'Number Paths (3)', 'Number Paths (4)', 'Pageranks (100)', 'Pageranks (1000)']
common_neighbors_acc = [55.51,	58.43,	54.99,	4.62]
edge_weights_acc = [58.24,	60.08,	56.55,	7.27]
num_paths_3_acc = [45.39,	58.3,	54.42,	4.13]
num_paths_4_acc = [55.62,	58.5,	54.96,	4.86]

a = [11.93,	50.94,	47.23,	3.56]
b = [11.93,	48.47,	46.59,	3.56]
c = [12.36,	51.71,	47.08,	3.59]

pageranks_100 = [(i+j+k)/3 for i,j,k in zip(a,b,c)]

aa = [34.68,	51.69,	48.38,	3.55]
bb = [34.18,	52.01,	48.52,	3.54]
cc = [34.48,	51.66,	48.62,	3.42]

pageranks_1000 = [(i+j+k)/3 for i,j,k in zip(aa,bb,cc)]

data = [common_neighbors_acc,edge_weights_acc,num_paths_3_acc,num_paths_4_acc,pageranks_100,pageranks_1000]
X = np.arange(4)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Accuracy values for different methods (Dataset 1)')
ax.set_ylabel('Percentage')

l0 = ax.bar(X+0.00,data[0],color = 'c',width = 0.15)

l1 = ax.bar(X+0.15,data[1],color = '#6e2aeb',width = 0.15)

l2 = ax.bar(X+0.30,data[2],color = '#ebb12a',width = 0.15)

l3 = ax.bar(X+0.45,data[3],color = '#eb2adb',width = 0.15)

l4 = ax.bar(X+0.60,data[4],color = '#e3e032',width = 0.15)

l5 = ax.bar(X+0.75,data[5],color = '#2dc25c',width = 0.15)

ax.legend((l0,l1,l2,l3,l4,l5),(test_types))

ax.set_xticks( np.arange(0,4)+.375)
ax.set_xticklabels(acc_types)
ax.set_yticks(np.arange(0,101,10))

# plt.show()
plt.savefig('imgs/dataset1.png')
