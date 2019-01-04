'''
    2018-12-15 20:00: Toulouse

    2019-01-04 20:00: Barcelona
'''
# import the 
import numpy as np
import random
from sklearn.utils import check_array

import sys
sys.path.append('./matrix_calculater')
sys.path.append('./snn_graph')
from matrix_calculater import MatrixCalculater
from snn_graph import SNNGraph

def snn(X, k=7, metric='minkowski', p=2, link_weight='unweighted', point_threshold=5, edge_threshold=3):
'''
    Performing SNN(Shared Nearest Neigbors) clustering algorithm

    Parameters:
    -----------
    X -> The dataset waiting for clustering
    k -> define the length of nearest neighbors list (default is 7)
    metric -> define the way to calculate the distance between two samples in the dataset
        (default is using the 'minkowski' method; there are also two ways to define the distance:
            cosine similarity -> dot(x1, x2)/ (sqrt(x1)*sqrt(x2))
        )
    p -> the number of norm used in the 'minkowski' distance calculation(just useful with the option of 
        metric = 'minkowski')
    link_weight -> define the way to calculate the weight of links in the SNN graph
    threshold -> define the miniuim standard for the weight of links existed
        (if weight of link < threshold, then delete this link in the SNN graph)

    Features:
    ---------
    core_samples -> The representative points chosen for SNN
    labels -> the labels for all samples in the dataset chosen
'''
    # juedge the k is greater than 0 or not
    if not (k.is_integer() or k <= 0):
        raise ValueError("K must be an integer greater than 0")
    
    # distance_calculater the k-nearest-neighbors list for each point
    distance_calculater = MatrixCalculater(k=k, metric=metric, p=p)
    # distance_calculater has the simi_array_all and simi_array_k
    distance_calculater.calculate(X)

    # snn_grapher is the object for construct the SNN graph
    snn_grapher = SNNGraph(link_weight=link_weight, point_threshold=point_threshold, edge_threshold=edge_threshold)
    # snn_grapher has the attributes after constructing with simi_array_k from distance_calculater
    #   1. graph -> original snn graph
    #   2. repr_points -> representative points list(index) filtered by point_threshold
    #   3. graph_filtered -> final snn graph after deleting the links
    snn_grapher.constructFinalGraph(distance_calculater.simi_array_k)

    # using the final snn graph for getting the clusters(labels_index)
    labels_index = _propagateCluster(snn_grapher.graph_filtered, snn_grapher.repr_points, range(len(X)))

    # return all informations ued
    return ()

def _propagateCluster(graph, repr_points, points):
    '''
        propagation the cluster with the graph_filtered and repr_points
        considering the representative point progated with other repr_points

        Parameters:
        -----------
        graph -> the SNN graph filtered(structure is [{}, {}, ...] )
        repr_points -> the index of representataive points in the graph 
        points -> all points index in the dataset(np.array(range(0 to size))

        Return:
        -------
        labels_index -> the index of point in each cluster label
    '''
    labels_index = []
    while repr_points.size > 0:
        # set the repr_points_used as []
        repr_points_used = np.array([])
        # choose one repr_point as the init randomly
        i_random = random.randint(0, repr_points.size-1)
        repr_point_random = repr_points[i_random]
        # using the queue for one repr_point
        repr_point_queue = np.array([repr_point_random])

        cluster = np.array([])
        while repr_point_queue.size > 0:
            first = repr_point_queue[0]
            np.delete(repr_point_queue, 0)

            np.append(repr_points_used, first)
            np.append(cluster, first)

            dict_neighbor_first = graph[first]
            if not dict_neighbor_first:
                # no neigbors, just as one cluster
                pass
            else:
                for key,value in dict_neighbor_first.items():
                    if key in repr_points:
                        np.append(repr_point_queue, key)
                        np.append(repr_points_used, key)
                    else:
                        np.append(cluster, key)
        # update the repr_points left
        repr_points = np.setdiff1d(repr_points, repr_points_used)
        # update the labels_index
        labels_index.append(cluster)
        # update the points left
        points = np.setdiff1d(points, cluster)
    # finally, add all points left in the points as the noise points, which has the label of 0
    labels_index.insert(0, points)
    return labels_index



class SNN(object):
    def __init__(self, k=7, metric='minkowski', p=2, link_weight='unweighted', point_threshold=5, edge_threshold=3):
        self.k = k
        self.metric = metric
        self.p = p
        self.link_weight = link_weight
        self.point_threshold = point_threshold
        self.edge_threshold = edge_threshold

    def fit(self):
        X = check_array(X, accept_sparse='csr')
        
