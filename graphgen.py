import matplotlib.pyplot as plt
import numpy as np
# right now this is just for dataset 1
acc_types = ['Total','White victory','Black victory','Draw']
test_types = ['Common Neighbors','Edge Weights', 'Number Paths (3)', 'Number Paths (4)']
common_neighbors_acc = [55.51,58.43,54.99,4.62]
edge_weights_acc = [58.24,60.08,56.55,7.27]
num_paths_3_acc = [45.39,58.3,54.42,4.13]
num_paths_4_acc = [57.36,60.11,56.77,3.73]

data = [common_neighbors_acc,edge_weights_acc,num_paths_3_acc,num_paths_4_acc]
X = np.arange(4)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Accurarcy values for different methods')
ax.set_ylabel('Percentage')

l0 = ax.bar(X+0.00,data[0],color = 'c',width = 0.20)

l1 = ax.bar(X+0.20,data[1],color = '#6e2aeb',width = 0.20)

l2 = ax.bar(X+0.40,data[2],color = '#ebb12a',width = 0.20)

l3 = ax.bar(X+0.60,data[3],color = '#eb2adb',width = 0.20)

ax.legend((l0,l1,l2,l3),(test_types))

ax.set_xticks( np.arange(-0.1,3.7,)+.4)
ax.set_xticklabels(acc_types)
ax.set_yticks(np.arange(0,101,10))

plt.show()
