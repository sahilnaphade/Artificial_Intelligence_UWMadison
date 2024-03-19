import csv
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


def load_data(filepath):
    all_data = []
    with open(filepath, newline='') as f:
        freader = csv.DictReader(f)
        for row in freader:
            all_data.append(dict(row))
    return all_data

def calc_features(row):
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    return np.array([x1, x2, x3, x4, x5, x6], dtype='float64')
    pass

def distance(feat_1, feat_2):
    temp = feat_1 - feat_2
    sum_eq = np.dot(temp.T, temp)
    return(np.sqrt(sum_eq))

def hac(features):
    total_feat = len(features)
    cluster_list = np.arange(total_feat).reshape(-1, 1).tolist()
    cluster_distance_matrix = np.zeros((total_feat, total_feat))
    clustered_countries = set()
    Z = np.zeros((total_feat - 1, 4))
    
    # Populate initial distances
    for ii in range(total_feat):
        for jj in range(ii, total_feat):
            cluster_distance_matrix[ii][jj] = cluster_distance_matrix[jj][ii] = np.linalg.norm(features[ii] - features[jj])

    # cluster_distance_matrix = np.array([[0, 17, 21, 31, 23], [17, 0, 30, 34, 21], [21, 30, 0 , 28, 39], [31, 34, 28, 0, 43], [23, 21, 39, 43, 0]], dtype=float)

    # print(cluster_distance_matrix)
    for clustering_idx in range(total_feat - 1):
        smallest_dist = float('inf')
        idx_1 = idx_2 = -1
        
        # Find the closest pair of clusters
        ii_len = jj_len = -1
        ii_idx = jj_idx = -1
        for ii in range(total_feat):
            for jj in range(ii+1, total_feat):
                # print(f"cluster_distance_matrix is {cluster_distance_matrix[ii][jj]} and smallest_dist is {smallest_dist}, {cluster_distance_matrix[ii][jj] < smallest_dist}")
                if cluster_distance_matrix[ii][jj] < smallest_dist:
                    smallest_dist = cluster_distance_matrix[ii][jj]
                    idx_1 = ii
                    idx_2 = jj
                # We consider each country to be its own cluster == we create a cluster of x+y
                # From the reverse side, check if one of the cluster is already part of another cluster
                for each_cluster in reversed(cluster_list):
                    if idx_1 in each_cluster:
                        # print(f"found the index {idx_1} in cluster {each_cluster}")
                        ii_len = len(each_cluster)
                        # print(f"ii_len is {ii_len}")
                        ii_idx = cluster_list.index(each_cluster)
                        break
                for each_cluster in reversed(cluster_list):
                    if idx_2 in each_cluster:
                        # print(f"found the index {idx_2} in cluster {each_cluster}")
                        jj_len = len(each_cluster)
                        # print(f"jj_len is {jj_len}")
                        jj_idx = cluster_list.index(each_cluster)
                        break
                if ii_idx == jj_idx:
                    continue

        # print("Merging the indices {} and {} with distance {}".format(idx_1, idx_2, smallest_dist))
        cluster_distance_matrix[idx_1][idx_2] = cluster_distance_matrix[idx_2][idx_1] = float('inf')
        # print(f"Cluster at iith index {ii_idx} is {cluster_list[ii_idx]} and cluster at jjth index {jj_idx} is {cluster_list[jj_idx]}")
        # print(f"the combination of these clusters is {cluster_list[ii_idx] + cluster_list[jj_idx]}")
        updated_cluster = cluster_list[ii_idx] + cluster_list[jj_idx]
        for ii in cluster_list[ii_idx]:
            for jj in cluster_list[jj_idx]:
                # Update the distance between the current cluster to infinite (We dont want to merge them again)
                cluster_distance_matrix[jj][ii] = cluster_distance_matrix[ii][jj] = float('inf')
        cluster_list.append(updated_cluster)

        # Update Z with merging information
        Z[clustering_idx, 0] = min(ii_idx, jj_idx)
        Z[clustering_idx, 1] = max(ii_idx, jj_idx)
        Z[clustering_idx, 2] = smallest_dist
        total_len = ii_len + jj_len
        # print(f"THE TOTAL LENGTH OF TH CLUSTER NOW IS {total_len}")
        Z[clustering_idx, 3] = total_len
        # print(cluster_distance_matrix)
        
        # Merge the clusters
        clustered_countries.add(idx_1)
        clustered_countries.add(idx_2)
        
        # Update the cluster distance matrix
        for updater in range(total_feat):
            for index_1 in cluster_list[ii_idx]:
                for index_2 in cluster_list[jj_idx]:
                    cluster_distance_matrix[index_1][updater] = cluster_distance_matrix[updater][index_1] = cluster_distance_matrix[index_2][updater] = cluster_distance_matrix[updater][index_2] = max(cluster_distance_matrix[index_1][updater], cluster_distance_matrix[index_2][updater])
        # print(f"At the end of iteration {clustering_idx}, the distance matrix is:")
        # print(cluster_distance_matrix)
    return Z

def fig_hac(Z, names):
    fig = plt.figure()
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.title('Dendrogram')
    plt.xlabel('Countries')
    plt.ylabel('Distance')
    return fig 

def normalize_features(features):
    means = np.mean(features, axis=0)
    std_dev = np.std(features, axis=0)
    return ((features - means) / std_dev).tolist()


# if __name__ == "__main__":
#     data = load_data(filepath="./countries.csv")
#     print(data)