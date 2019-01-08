'''
    2018-12-15 20:00: Toulouse

    2019-01-04 20:00: Barcelona
'''
# import the standard libraries
import numpy as np
import random
from sklearn.utils import check_array
import matplotlib.pyplot as plt

# import the classes written
import sys
sys.path.append('./matrix_calculater')
sys.path.append('./snn_graph')
from matrix_calculater import MatrixCalculater
from snn_graph import SNNGraph

def snn(X, k=7, metric='minkowski', p=2, link_weight='unweighted', point_threshold=12, edge_threshold=3):
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
        labels_ -> The array of label for each point
        graph_final -> The final snn graph constructed
        repr_points -> The representative points chosen for SNN
        graph_original -> The original snn graph constructed
        simi_array_k -> The index of k-nearest-neighbors list for each points in the dataset 
    '''
    # juedge the k is greater than 0 or not
    if not (isinstance(k, int) or k <= 0):
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

    # transforming the labels_index into the array of label for each points
    labels_ = _getLabels(labels_index, len(X))

    # return all informations ued
    return (labels_, snn_grapher.graph_filtered, snn_grapher.repr_points, snn_grapher.graph, distance_calculater.simi_array_k)

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
    print("Enter the propagate!")
    labels_index = []
    while repr_points.size > 0:
        # set the repr_points_used as []
        repr_points_used = np.array([])
        # choose one repr_point as the init randomly
        i_random = random.randint(0, repr_points.size-1)
        repr_point_random = repr_points[i_random]
        # using the queue for one repr_point
        repr_point_queue = np.array([repr_point_random])

        cluster = np.array([], dtype = int)
        while repr_point_queue.size > 0:
            first = repr_point_queue[0]
            repr_point_queue = np.delete(repr_point_queue, 0)

            repr_points_used = np.append(repr_points_used, first)
            cluster = np.append(cluster, first)
            dict_neighbor_first = graph[first]
            if not dict_neighbor_first:
                # no neigbors, just as one cluster
                continue
            else:
                for key,value in dict_neighbor_first.items():
                    if key in repr_points:
                        if not key in repr_point_queue and not key in repr_points_used:
                            repr_point_queue = np.append(repr_point_queue, key)
                        else:
                            continue
                    elif not key in cluster and key in points:
                        cluster = np.append(cluster, key)
                    else:
                        continue
        # update the repr_points left
        repr_points = np.setdiff1d(repr_points, repr_points_used)
        # update the labels_index
        labels_index.append(cluster)
        # update the points left
        points = np.setdiff1d(points, cluster)

    # finally, add all points left in the points as the noise points, which has the label of 0
    labels_index.insert(0, points)
    return labels_index

def _getLabels(labels_index, data_len):
    '''
        transforming the labels_index into the array of label for each point
        noise_points to be set label -1
        others are set as postive interger(from 0 to labels type)

        Parameters:
        -----------
        labels_index -> the index of points in each cluster
        data_len -> the len of the data

        Return:
        -------
        labels_ -> the array of label for each point(len(points))
            noise_point = -1
            others from 0 to len(labels_index)-1
    '''
    labels_ = []
    # initializing the labels_ into all items to be -2, which means the impossible label
    for i in range(data_len):
        labels_.append(-2)

    for l_index in range(len(labels_index)):
        for p_index in labels_index[l_index]:
            if labels_[p_index] == -2:
                labels_[p_index] = l_index -1
            else:
                #print("Wrong, the label before of ", p_index, " is ", labels_[p_index], ", new is ", l_index-1)
                raise ValueError("There are two different labels for the same point!")
    
    return labels_


class SNN(object):
    def __init__(self, k=7, metric='minkowski', p=2, link_weight='unweighted', point_threshold=12, edge_threshold=3):
        self.k = k
        self.metric = metric
        self.p = p
        self.link_weight = link_weight
        self.point_threshold = point_threshold
        self.edge_threshold = edge_threshold
    

    # def _drawGraph2d(X, graph):
    #     '''
    #         The common function of drawing a graph, where the structure of graph is
    #         [{}, {}, ....](The array of dict)
    #         Here for simpility, the function can just for points in 2d

    #         Parameter:
    #         ----------
    #         points -> the array of points in 2d
    #         graph -> the graph to be drawed
    #     '''
    #     # draw the points before adding the edges


    def fit(self, X):
        X = check_array(X, accept_sparse='csr')
        (self.labels_, self.graph_final, self.repr_points, 
            self.graph_original, self.simi_array_k) = snn(X, 
            k=self.k, 
            metric=self.metric, 
            p=self.p, 
            link_weight=self.link_weight, 
            point_threshold=self.point_threshold, 
            edge_threshold=self.edge_threshold)
        
    def showRepresentativePoints2d(self, X):
        '''
            Showing the representative points with the dataset(just in 2d)
        '''
        # using the scatter of matplot for visualing the represntative points
        if self.repr_points is None:
            raise TypeError("Should filter the representative points before showing the repr_points!")
        else:
            # repr_points is the array of index of repr point
            repr_x = []
            repr_y = []
            for index in self.repr_points:
                repr_x.append(X[index][0])
                repr_y.append(X[index][1])
            
            plt.scatter(repr_x, repr_y, c='blue')
            plt.title('Graph of showing the represenative points')
            plt.xlabel('axis x')
            plt.ylabel('axis y')
            plt.show()

    # def showOriginalGraph2d(self, X):
    #     '''
    #         draw the original SNN graph(just in 2d)
    #     '''
    #     if self.graph is None:
    #         raise TypeError("Should construct the graph before show SNN graph")
    #     else:
    #         # draw the SNN original graph
    #         pass

    # def showFinalGraph2d(self, X):
    #     pass