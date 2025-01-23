# %%
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.plotAnomaly as plotA
import utils.Triangulation as tri
import esda
from Model import EIF


sampleSize = 20
data_paths_tif, data_paths_geojson, data_paths_geojson_zipped = utils.collectFiles(sampleSize)# .collectFiles() # this will automatically give 20
num = 0
myData = utils.engineer(num, 
                              data_paths_tif, 
                              data_paths_geojson, 
                              data_paths_geojson_zipped,
							  False)
data = myData.data.copy(deep=True)
delineations = myData.delineations.copy(deep=True)
mask = myData.mask.copy(deep=True)
spectralData = myData.spectralData
erf_num = myData.erf
refData = myData.ref_data.copy(deep=True)
# For plotting
tryout = spectralData["rgb"][0:3].rio.clip(mask.geometry.values, mask.crs, drop=True, invert=False)
tryout = tryout/255


# %%
from sklearn.model_selection import train_test_split

y = np.array(data.loc[:, "Y"]).T 
    # Change outlier to 1 and inlier to 0 in data
y = np.where(y == 'Outlier', 1, 0)

outliers_fraction = np.count_nonzero(y) / len(y)

X = np.array(data.loc[:, "confidence":]) 

# 60% data for training and 40% for testing
X_train, X_test, _, y_test = train_test_split(X,
                                            y,
                                            test_size=0.4,
                                            stratify=y,
                                            random_state=42)

# %%
# standardizing data for processing
X_train_norm = utils.engineer._scaleData(X_train)
X_test_norm = utils.engineer._scaleData(X_test)


# %%
clf = EIF(outliers_fraction, data.loc[:, "confidence":].columns)

# %%
clf.fit(X_train_norm)

# %%
y_pred = clf.decision_function(X_test_norm)

# %%

from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_true=y_test, 
                                 y_pred=y_pred)#np.array(clf.decision_scores_))
plt.show()


# %%
# Compute ROC curve and ROC area
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)
print(f'Average precision-recall score: {average_precision:.2f}')

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('foo.png')

# %%




# %%



from Model import Geary

X = utils.engineer._scaleData(X)
clf = Geary(contamination=outliers_fraction, 
            geometry=data["geometry"], 
            centroid=data["centroid"])
clf.fit(X)

test_scores = clf.decision_scores_
labels = clf.labels_


# %%
og_data = data.copy(deep=True)
fin_data = list(og_data.loc[:,'geometry':'roundness'].columns) + ["compactness", "convexity", "bendingE",  "DSM" , "NDRE", "OSAVI", "ASM", "Corr"]+ list(og_data.loc[:,"z0":'z24'])
og_data = og_data.loc[:, fin_data]
data = og_data.copy(deep=True)
data.loc[:,'confidence':] = utils.engineer._scaleData(data.loc[:, "confidence":])
clf = EIF(outliers_fraction, data.loc[:, "confidence":].columns)
clf.fit(X)
test_scores = clf.decision_scores_
labels = clf.labels_

# %%

normal = data[labels == 0]
abnormal = data[labels == 1]

plotA.plot(tryout, normal, abnormal, 'foo2.png')
# fig, ax = plt.subplots(figsize=(15, 15))
# tryout.plot.imshow(ax=ax)
# normal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
# abnormal.plot(ax=ax, facecolor = 'none',edgecolor='blue')
# plt.savefig('foo.png')


# %%

# just want to see if ABOD is indeed too slow
# default is way too slow!
from pyod.models.lof import LOF

clf = LOF(contamination=outliers_fraction)
clf.fit(data.loc[:, "confidence":])
test_scores = clf.decision_scores_
labels = clf.labels_

normal = data[labels == 0]
abnormal = data[labels == 1]

plotA.plot(tryout, normal, abnormal, 'foo2.png')

# %%

# this is how we use PCA with reconstruction

from pyod.models.kpca import KPCA

clf = KPCA(contamination=outliers_fraction, kernel='linear')
clf.fit(X)
test_scores = clf.decision_scores_
labels = clf.labels_

normal = data[labels == 0]
abnormal = data[labels == 1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
normal.plot(ax=ax, facecolor = 'none',edgecolor='red')
abnormal.plot(ax=ax, facecolor = 'none',edgecolor='blue')
plt.savefig('foo3.png')

# %%
# TODO: Delete because it doesn't work
# from pyod.models.cblof import CBLOF
# from sklearn.cluster import HDBSCAN

# clf = CBLOF(contamination=outliers_fraction, clustering_estimator=HDBSCAN(min_cluster_size=15))
# clf.fit(X)
# test_scores = clf.decision_scores_
# labels = clf.labels_

# normal = data[labels == 0]
# abnormal = data[labels == 1]

# fig, ax = plt.subplots(figsize=(15, 15))
# tryout.plot.imshow(ax=ax)
# normal.plot(ax=ax, facecolor = 'none',edgecolor='red') 
# abnormal.plot(ax=ax, facecolor = 'none',edgecolor='blue')
# plt.savefig('foo2.png')



# %%
# TODO: https://github.com/jhmadsen/DDoutlier/blob/master/R/OutlierFunctionLibrary.R
#       Might not need to include thi just becuase it is not a popular method at all
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

def RDOS(dataset, k=5, h=1):
    dataset = np.array(dataset)
    n, d = dataset.shape

    if not isinstance(k, int):
        raise ValueError('k input must be numeric')
    if k >= n or k < 1:
        raise ValueError('k input must be less than number of observations and greater than 0')
    if not isinstance(h, (int, float)):
        raise ValueError('h input must be numeric')
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError('dataset input is not numeric')

    dist_matrix = squareform(pdist(dataset))
    nbrs = NearestNeighbors(n_neighbors=k).fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)

    def func_dist(x1, x2):
        return len(np.intersect1d(x1, x2))

    sNN_matrix = np.array([[func_dist(indices[i], indices[j]) for j in range(n)] for i in range(n)])

    neighborhood = []
    for i in range(n):
        kNN = indices[i]
        rNN = np.where(indices == i)[0]
        sNN = np.where(sNN_matrix[i] > 0)[0]
        neighborhood.append(np.unique(np.concatenate((kNN, rNN, sNN))))

    px = np.zeros(n)
    for i in range(n):
        Kgaussian = (1 / ((2 * np.pi) ** (d / 2))) * np.exp(-(dist_matrix[i, neighborhood[i]] / (2 * h ** 2)))
        px[i] = (1 / (len(neighborhood[i]) + 1)) * np.sum((1 / (h ** d)) * Kgaussian)

    RDOS = np.zeros(n)
    for i in range(n):
        RDOS[i] = np.sum(px[neighborhood[i]]) / (len(neighborhood[i]) * px[i])

    return RDOS

a = RDOS(X, k=5, h=1)

_, ax = plt.subplots(1, figsize=(20, 20))
tryout.plot.imshow(ax=ax)
data.assign(cl= a).plot(column='cl', categorical=False,
        k=5, cmap='viridis', linewidth=0.1, ax=ax,
        edgecolor='white', legend=True, alpha=0.7)
ax.set_axis_off()
plt.title("Anomaly Scores")
plt.savefig('foo3.png')




# %%

# https://github.com/ValaryLim/pyODPlus/blob/main/outlier_detection/rocf.py
# import packages
import numpy as np
from scipy.spatial.distance import cityblock, euclidean # distance metrics
from heapq import heappush, heappop # priority queue
from queue import Queue # queue
import matplotlib.pyplot as plt # graph

class ROCF():
    def __init__(self, distance_metric="euclidean", k=3, threshold=0.1):
        '''
        Parameters
        ----------
        distance_metric : str in ("manhattan", "euclidean"), optional 
            (default="euclidean")
            The distance metric to use to compute k nearest neighbours.
        
        k : int, optional (default=3)
            k number of nearest neigbours used to form MUtual Neighbour Graph.
        
        threshold : float, optional (default=0.1)
            Threshold set for any cluster to be considered an outlier. 
            Each cluster has an ROCF value.
            If max({ROCF}) < threshold, no cluster is considered as outlier.
            Else, all clusters with smaller size than cluster with max ROCF are
            tagged as outliers. 
        '''
        # checks for input validity
        if distance_metric not in ["euclidean", "manhattan"]:
            raise ValueError("Invalid distance_metric input. Only accepts 'euclidean' or 'manhattan'.")
        
        try:
            if int(k) != k:
                raise ValueError("Invalid k input. k should be an integer")
        except: 
            raise ValueError("Invalid k input. k should be an integer")
        
        try: 
            if float(threshold) != threshold or threshold < 0 or threshold > 1:
                raise ValueError("Invalid threshold input. threshold should be a float between 0 and 1")
        except:
            raise ValueError("Invalid threshold input. threshold should be a float between 0 and 1")


        # initialise input attributes
        self.distance_metric = distance_metric
        self.k = int(k)
        self.threshold = threshold
        
        # define computed attributes
        self.outliers = None
        self.transition_levels = None
        self.rocfs = None
        self.cluster_labels = None
        self.cluster_groups = None
        self.k_nearest_neighbours = None

    def fit(self, X):
        '''
        Runs ROCF algorithm to detect outliers.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        self : object
            Fitted estimator.
        '''
        # retrieve k nearest neighbours
        k_nearest_neighbours = self._retrieve_k_nearest_neighbours(X)

        # cluster datasets using mutual neighbour graph
        cluster_labels, cluster_groups = self._retrieve_clusters_mung(X, k_nearest_neighbours)

        # compute outliers
        outliers, transition_levels, rocfs, cluster_groups_sorted = self._compute_outliers_rocf(X, cluster_groups)

        # update self object
        self.k_nearest_neighbours = k_nearest_neighbours
        self.cluster_labels = cluster_labels
        self.cluster_groups = cluster_groups_sorted
        self.outliers = outliers
        self.transition_levels = transition_levels
        self.rocfs = rocfs

        return self
 
    def _compute_distance(self, v1, v2):
        '''
        Computes distance between two data points
        
        Parameters
        ----------
        v1 : numpy array of shape (n_features,)
            The first data point.
        
        v2 : numpy array of shape (n_features,)
            The second data point

        Returns
        ----------
        distance : float
            Distance between two input data points.
        '''
        distance_metric = self.get_distance_metric()
        if distance_metric == "euclidean":
            return euclidean(v1, v2)
        else:
            return cityblock(v1, v2)
    
    def _retrieve_k_nearest_neighbours(self, X):
        '''
        Retrieves k nearest neighbours
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        k_nearest_neighbours: numpy array of shape (n_samples, k_clusters)
        '''
        k_nearest_neighbours = []
        k = self.get_k()

        # iterate through each element in X and find the k nearest neighbours
        for i in range(len(X)):
            # create heap to store k nearest neighbours
            neighbours_i = []
            
            # iterate through all other points
            for j in range(len(X)):
                if i == j:
                    continue # do not count itself
                
                # compute distance between i and j
                current_dist = self._compute_distance(X[i], X[j])
                
                if len(neighbours_i) < k:
                    # insufficient neighbours, add into neighbour heap
                    heappush(neighbours_i, (-current_dist, j))
                
                if len(neighbours_i) == k: 
                    # retrieve largest distance neighbour
                    largest_neighbour = heappop(neighbours_i)
                    
                    if current_dist < -largest_neighbour[0]:
                        # distance between i and j is smaller than largest neighbour
                        # update neighbours to include current element
                        heappush(neighbours_i, (-current_dist, j))
                    else:
                        # replace largest element back in neighbours heap
                        heappush(neighbours_i, largest_neighbour)
                        
            # extract the neighbours
            neighbours_i = set([x[1] for x in neighbours_i])
                    
            k_nearest_neighbours.append(neighbours_i)

        return k_nearest_neighbours

    def _retrieve_clusters_mung(self, X, k_nearest_neighbours):
        '''
        Retrieves clusters using the MUtual Neighbour Graph (MUNG) algorithm
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        k_nearest_neighbours : numpy array of shape (n_samples, k_clusters)
            The k nearest neighbours of each sample.
        
        Returns
        ----------
        cluster_labels : numpy array of shape (n_samples,)
            Array indicating the cluster that each sample is classified as
        
        cluster_groups : numpy array of shape (n_clusters, 2)
            Each sample row contains (size of cluster, samples in cluster) 
        '''
        # define visited
        visited = [False] * len(X)

        # define clusters for each point (initialise to -1)
        cluster_labels = [-1] * len(X)

        # define cluster groups
        cluster_groups = []

        label = 0 # counter for cluster label

        # iterate through each element 
        for i in range(len(X)):
            if visited[i]: # ignore if element is already visited
                continue
        
            # define queue to store mutual neighbours of cluster
            mutual_neighbours = Queue()
            mutual_neighbours.put(i)
            visited[i] = True
            cluster_labels[i] = label

            # define cluster group
            current_cluster = set()
            current_cluster.add(i)

            # while there still exists mutual neighbours
            while not mutual_neighbours.empty():
                # retrieve next mutual neighbour
                v = mutual_neighbours.get()

                # find all unvisited mutual neighbours of v
                for v_neighbours in k_nearest_neighbours[v]:
                    if (v in k_nearest_neighbours[v_neighbours]) and (not visited[v_neighbours]):
                        # mark as visited, label cluster group
                        visited[v_neighbours] = True
                        cluster_labels[v_neighbours] = label
                        current_cluster.add(v_neighbours)

                        # if v is a mutual neighbour and is not visited
                        mutual_neighbours.put(v_neighbours)
            
            # update cluster group
            cluster_groups.append([len(current_cluster), current_cluster])

            label += 1 # increment label

        return np.array(cluster_labels), np.array(cluster_groups)
    
    def _compute_outliers_rocf(self, X, cluster_groups):
        '''
        Computes outliers using the ROCF method
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        cluster_groups : numpy array of shape (n_clusters, 2)
            Each sample row contains (size of cluster, samples in cluster) 
        
        Returns
        ----------
        outliers : numpy array of shape (n_samples,)
            Array indicating if sample is outlier (1) or normal (0)
        
        transition_levels : numpy array of shape (n_clusters - 1,)
            Transition level of each cluster

        rocfs : numpy array of shape (n_clusters - 1,)
            ROCF of each cluster
        '''
        # define outliers
        outliers = [0] * len(X) # 0 marks normal data point

        # filter out all cluster groups smaller than k
        k = self.get_k()

        # points in clusters of size < k are outleirs
        for cg in cluster_groups:
            if self._get_cluster_size(cg) < k:
                # cluster is an outlier cluster
                cg_points = self._get_cluster_points(cg)
                for v in cg_points:
                    # tag all points in cluster as outliers
                    outliers[v] = 1 
        
        # sort clusters
        cluster_groups_sorted = sorted(cluster_groups, key=lambda x: self._get_cluster_size(x), reverse=False)

        # compute transition levels and rocf
        transition_levels = []
        rocfs = [] 
        # iterate through cluster groups
        for i in range(len(cluster_groups_sorted) - 1):
            c1_size = self._get_cluster_size(cluster_groups_sorted[i])
            c2_size = self._get_cluster_size(cluster_groups_sorted[i+1])
            tl = c2_size / c1_size
            rocf = 1 - np.exp(-tl / c1_size)

            # update transition levels and rocfs
            transition_levels.append(tl)
            rocfs.append(rocf)
        
        # retrieve maximum rocfs
        max_rocf = max(rocfs)
        max_rocf_index = max(ind for ind, value in enumerate(rocfs) if value == max_rocf)

        # identify outliers from maximum rocfs
        threshold = self.get_threshold()
        if max_rocf > threshold: # if greater than threshold, some clusters are outliers
            for i in range(max_rocf_index): 
                for v in self._get_cluster_points(cluster_groups_sorted[i]):
                    outliers[v] = 1 # tag points in outlier clusters
        
        return np.array(outliers), np.array(transition_levels), np.array(rocfs), np.array(cluster_groups_sorted)

    def get_outliers(self):
        return self.outliers
    
    def get_k_nearest_neighbours(self):
        return self.k_nearest_neighbours
    
    def get_cluster_labels(self):
        return self.cluster_labels
    
    def get_cluster_groups(self):
        return self.cluster_groups

    def get_outlier_rate(self):
        return sum(self.outliers) / len(self.outliers)
    
    def get_transition_levels(self):
        return self.transition_levels
    
    def get_rocfs(self):
        return self.rocfs
    
    def _get_cluster_size(self, cluster):
        return cluster[0]
    
    def _get_cluster_points(self, cluster):
        return cluster[1]
    
    def get_k(self):
        return self.k
    
    def get_threshold(self):
        return self.threshold
    
    def get_distance_metric(self):
        return self.distance_metric

    def plot_decision_graph(self):
        # retrieve rocf
        rocfs = self.get_rocfs()

        # plot graph
        plt.figure(figsize=(20,10))
        plt.scatter([i for i in range(len(rocfs))], rocfs)
        plt.show()


clf = ROCF(distance_metric="euclidean", k=30, threshold=0.1)
clf.fit(X)

outliers = clf.get_outliers()
normal = data[outliers == 0]
abnormal = data[outliers == 1]

fig, ax = plt.subplots(figsize=(15, 15))
tryout.plot.imshow(ax=ax)
normal.plot(ax=ax, facecolor='none', edgecolor='red')
abnormal.plot(ax=ax, facecolor='none', edgecolor='blue')
plt.savefig('foo4.png')













# %%
# Grouping of anomalies into individual categories!

# %%
# For Correction purposes
    # False-positives
        # Remove observation
    # Over-segmentations   
        # TODO: https://medium.com/@jesse419419/understanding-iou-and-nms-by-a-j-dcebaad60652
        #       https://infoscience.epfl.ch/server/api/core/bitstreams/768fc3bc-f7d4-4533-825d-5c398995526d/content
    # Under-segmentations
        # https://arxiv.org/pdf/2202.08682
        # https://www.mdpi.com/2072-4292/12/5/767
    # False-negatives - https://www.mdpi.com/2072-4292/11/4/410
        # YOlOv10
        # Can train YOLO on all available orchards
        # following this we can use it as a trianed model.
        # we then isolate the area that has a potential false-negative and test using trained YOLOv10.

    # Can build probability map using - https://www.mdpi.com/2072-4292/12/5/767