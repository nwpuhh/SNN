import sys
sys.path.append('./snn_graph')
sys.path.append('./matrix_calculater')
sys.path.append('./snn_')
from matrix_calculater import MatrixCalculater
from snn_graph import SNNGraph
from snn_ import SNN
import numpy as np
import matplotlib.pyplot as plt

# construct the test-dataset
data_test = [[0, 0], [1, 0], [0, 1], [1, 1], [-1, 0],
             [-1, -1], [0, -1], [3, 4], [3, 3], [5, 4], 
             [5, 5], [6, 6], [6, 5], [-10, -10], [-10, -11],
             [-10, -9]]
# the matrix calculater with default setting
# m = MatrixCalculater()
# m.calculate(data_test)
# print("Simi_array_k is ", m.simi_array_k)

# # construct the snn grapher with default setting
# snn_grapher = SNNGraph()
# snn_grapher.constructOrignialGraph(m.simi_array_k)
# print("The original graph is ", snn_grapher.graph)

# # choose the repr_points in the dataset
# snn_grapher.selectRepressentativePoints()
# print("The repr_points of dataset is ", snn_grapher.repr_points)

# # deleting the links less than the thresold
# snn_grapher.filterLinks()
# print("The final graph is ", snn_grapher.graph_filtered)

snn_model = SNN()
#snn_model.fit(data_test) 

# using the dataset from professor to show the result
data1 = np.genfromtxt("/home/nwpuhh/Workplace/INSA_TPs/MachineLearning/Unsupervised Learning/cham-data/t4.8k.dat", delimiter=" ", skip_header=1)
print("------------------")
snn_model.fit(data1)
snn_model.showRepresentativePoints2d(data1)
# print("The graph of data1 is ", snn_model.graph_final)
# print(5 in snn_model.repr_points)
# print("The label of data1 is ", snn_model.labels_)

# loss_index = np.array([])
# for i in range(len(data1)):
#     flag = False
#     for item in snn_model.labels_:
#         if i in item:
#             loss_index = np.append(loss_index, i)

# cmp(loss_index.tolist(), range(len(data1)))   

# print(loss_index)

def plot_model(model, data):
    labels = model.labels_
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.repr_points] = True
    # plot the clustering result and then return the score of clustering(silhouette_score)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plt.title('Estimated number of clusters: %d' % len(set(labels)))
    plt.show()

plot_model(snn_model, data1)
