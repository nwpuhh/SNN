import numpy as np 

class SNNGraph(object):
    '''
        Performing the SNN graph construction and plotting
        Step of graph performing
        1. construct the SNN graph orignial from the k-nearest neigbhors 
            list for each point: judge the edges existances, calculate 
            the weight of edges existed.
        
        2. calculate the weights for each point, judge the representative 
            points by filtering those points greater then point_threshold
        
        3. remove all edges having weights smaller than edge_threshold

        4. judge all points in which cluster, there are three key ideas:
            1) the representative point can be a new clutser
            2) all points connected with representative will be labeled with same cluster
            3) not representative point, no connection between representative points are noise points
    
        Parameters:
        -----------
        simi_array_k: the arrat of k nearest neighbors for each point
        link_weight: the way of calculating the weight for each link
            2 options: 'unweighted' -> calculate the sum of shared neighbors
                        'weighted' -> using the used equation for calculating the weight
        point_threshold: the threshold for choosing the representative points
        edge_threshold: the threshold for deleting links

        Features:
        ---------
        graph -> orignial snn graph constructed by the k-nearest-neigbors array
        repr_points -> array of representative points which are filtered after point_threshold
        graph_filtered -> snn graph after deleting the links less than edge_threshold
    '''
    def __init__(self, link_weight='unweighted', point_threshold=5, edge_threshold=3):
        self.link_weihted = link_weight
        self.point_threshold = point_threshold
        self.edge_threshold = edge_threshold
        self.graph = None
        self.repr_points = None
        self.graph_filtered = None

    def __calculateWeight(self, first_array_k, second_array_k):
        '''
            two methods for calculating the weight of a link
                'unweighted': number of shared neighbors
                'weighted': sum((k+1-m)*(k+1-n)), where first[m] == second[n]
        '''
        # calculate the number of shared neighbors in simi_array_k
        if self.link_weihted == 'unweighted':
            return len(np.intersect1d(first_array_k, second_array_k))
        elif self.link_weihted == 'weighted':
            common_neighbors = np.intersect1d(first_array_k, second_array_k)
            k = len(first_array_k)
            weight = 0
            for item in common_neighbors:
                m = np.where(first_array_k == item)
                n = np.where(second_array_k == item)
                weight = weight + (k+1-m)*(k+1-n)
            return weight
        else:
            raise ValueError("link_weighted must be unweighted or weighted!")    

    def __calculatePointWeightSum(self):
        '''
            Calculating the sum of weight for each point
            The sum of each point is the sum of weight of all edges which connect the point

            Return:
            -------
            points_weight -> the array for the weight of all points
        '''
        if self.graph is None:
            raise TypeError("Should construct the graph before calculate the weight for each point!")
        else:
            # using the self.graph for getting the sum of weight for each point
            # iterate the {key:value} in the dict
            points_weight = np.array([])
            for item in self.graph:
                p_w_temp = 0
                for _,value in item.items():
                    p_w_temp = p_w_temp + value
                np.append(points_weight, p_w_temp)
            
            return points_weight

    def constructOrignialGraph(self, simi_array_k):
        '''
            Using the simi_array_k for constructing the graph by changing the 
            graph -> ! Attention, this function will construct the original 
            snn graph, without choosing the representative points and deleting edges

            Features:
            ---------
            graph -> the ensemble of points with edges in the graph
        '''
        # graph is the ensemble of points in the graph
        # for each point, with the structure of [{p_link_index: value}]
        # which is the array of dictory to represent the snn graph
        graph = np.array([])
        for i in range(len(simi_array_k)):
            graph = np.append(graph, {})
        # construct the snn graph with the warning of size of simi_array_k
        for p_index in range(len(simi_array_k)):
            k = len(simi_array_k[0])
            if not(k == len(simi_array_k[p_index])):
                raise ValueError("The simi_array_k should have the same size!")
            else:
                for n_index in range(k):
                    # points are neighbors of each other, so add the link
                    if(n_index in simi_array_k[n_index]):
                        l_weight = self.__calculateWeight(simi_array_k[p_index], simi_array_k[n_index])
                        # add two key-value in dict of two points
                        graph[p_index][n_index] = l_weight
                        graph[n_index][p_index] = l_weight
        self.graph = graph

    def selectRepressentativePoints(self):
        '''
            selection of repressentative points by point_threshold
            The represenatative points will be stored as the index of point in the dataset

            Feature:
            --------
            repr_points -> the array of index of representative points in the dataset
        '''
        points_weight = self.__calculatePointWeightSum()
        repr_points = np.array([])
        for p_index in range(len(points_weight)):
            if points_weight[p_index] >= self.point_threshold:
                np.append(repr_points, points_weight[p_index])
        self.repr_points = repr_points

    def filterLinks(self):
        '''
            filtering the links by the edge_thrsold
            The new graph will be stored as graph_filtered

            Feature:
            --------
            graph_filtered -> the graph filtered by the edges(same structure as the graph in the feature)
        '''
        if self.graph is None:
            raise TypeError("Should construct the original graph before filtering the edges!")
        else:
            graph = self.graph
            for item in graph:
                for key,value in item.items():
                    # means that this edge should be deleted
                    if value < self.point_threshold:
                        del item[key]
            self.graph_filtered = graph

    def constructFinalGraph(self, simi_array_k):
        '''
            combining all functions in constructing the final SNN graph, which is
            the API for the class of snn_graph
        '''
        self.constructOrignialGraph(simi_array_k)
        self.selectRepressentativePoints()
        self.filterLinks()

    def showOriginalGraph2d(self, points):
        '''
            draw the original SNN graph(just in 2d)
        '''
        if self.graph is None:
            raise TypeError("Should construct the graph before show SNN graph")
        else:
            # draw the SNN original graph

    def showRepresentativePoints2d(self, points):
        '''
            Showing the representative points with the dataset(just in 2d)
        '''


    def showFinalGraph2d(self, points):