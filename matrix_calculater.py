from scipy.spatial.distance import pdist 
import numpy as np

class MatrixCalculater(object):
    '''
        Performing the Matrix of similarity of SNN calculation

        Parameters:
        -----------
        k -> the number of nearest neighbors chosen
        metric -> the way of calculating the similarity between two samples
        p -> the number of norm used in the 'minkowski' distance

        Features:
        ---------
        simi_array_all -> the sparse array for similartity between all different samples
            (here just use the normal calculation(O(n^2)), the improvement can used k-d tree)
        semi_array_k -> the sparse array for index of k-nearest similarity
    '''
    
    def __init__(self, k=7, metric='minkowski', p=2):
        self.k = k
        self.metric = metric
        self.p = p
    
    def __changeDistancesIntoMatrix(self, distances, point_num):
        '''
            private function:
            Performing the transformation of distances from pdist to distances array for each point

            ReturnValue:
            ------------
            (simi_array_all, simi_array_k)
        '''
        index_split = np.array([], dtype=int)
        cnt = 0
        for i in range(point_num-1, 0, -1):
            index_split = np.append(index_split, cnt+i)
            cnt = cnt + i 
        simi_array_all_lack = np.split(distances, index_split)
        # now the simi_array_all_lack lacks the rest of distance between points before
        # for ith element, it lacks i distance element
        for i in range(point_num):
            # reverse the distance array and add the 0 for d_i_i
            temp_distances = simi_array_all_lack[i][::-1]
            temp_distances = np.append(temp_distances, 0)
            
            # then add the lacked items
            for j in range(i-1, -1, -1):
                temp_distances = np.append(temp_distances, simi_array_all_lack[j][i])
            # get the reverse of distance array
            simi_array_all_lack[i] = temp_distances[::-1]
        
        # get the full simi_array
        simi_array_all = simi_array_all_lack

        # add the first array into the array_k : deleting the first of this because of 0
        k_nearest_neighbor_index = simi_array_all[0].argsort()[1:self.k+1]
        simi_array_k = np.array([k_nearest_neighbor_index, ])
        for i in range(1, point_num):
            k_nearest_neighbor_index = simi_array_all[i].argsort()[1:self.k+1]
            simi_array_k = np.concatenate((simi_array_k, np.array([k_nearest_neighbor_index])), axis=0)
        return (simi_array_all, simi_array_k)


    def calculate(self, X):
        '''
            Performing the calculation of similartity matrix of samples set X

            Parameters:
            -----------
            X -> the samples set
        '''
        # judge the method used for calculating the distance
        if self.metric == 'minkowski':
            # get the distance array of minkowski with chosen norm
            distances = pdist(X, self.metric, p=self.p)
            self.simi_array_all, self.simi_array_k = self.__changeDistancesIntoMatrix(distances, len(X))
        elif self.metric == 'cosine':
            distances = 1-pdist(X, self.metric)
            self.simi_array_all, self.simi_array_k = self.__changeDistancesIntoMatrix(distances, len(X))
        else:
            raise ValueError("metric must choose cosine or minkowski!")           
