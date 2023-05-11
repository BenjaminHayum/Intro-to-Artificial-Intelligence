import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage


def load_data(filepath):
    list_dictionaries = list()
    with open(filepath) as csv:
        for line in csv:
            curr_dictionary = dict()
            row_entries = line.split(",")
            if row_entries[0] != "#":
                curr_dictionary["HP"] = row_entries[5]
                curr_dictionary["Attack"] = row_entries[6]
                curr_dictionary["Defense"] = row_entries[7]
                curr_dictionary["Sp. Atk"] = row_entries[8]
                curr_dictionary["Sp. Def"] = row_entries[9]
                curr_dictionary["Speed"] = row_entries[10]
                list_dictionaries.append(curr_dictionary)
    return list_dictionaries


def calc_features(row):
    feature_representation = np.zeros(6, dtype=np.int64)
    feature_representation[0] = row["Attack"]
    feature_representation[1] = row["Sp. Atk"]
    feature_representation[2] = row["Speed"]
    feature_representation[3] = row["Defense"]
    feature_representation[4] = row["Sp. Def"]
    feature_representation[5] = row["HP"]
    return feature_representation


class cluster:
    def __init__(self, indices, points, linkage_distance, cluster_index):
        self.indices = indices
        self.points = points
        self.linkage_distance = linkage_distance
        self.cluster_index = cluster_index


def hac(features):
    # Initializing Z
    Z = np.zeros((len(features) - 1, 4))

    # Creating the original distance matrix
    distance_matrix = np.zeros((len(features), len(features)))
    for i in range(len(features)):
        i_point = features[i]
        for j in range(len(features)):
            j_point = features[j]
            distance_matrix[i, j] = np.linalg.norm(i_point - j_point)
        distance_matrix[i, i] = np.inf
    # List that will be updated to hold all of the most recent clusters
    cluster_list = list()
    for n in range(len(features)):
        curr_cluster = cluster(list([n]), list([features[n]]), None, int(n))
        cluster_list.append(curr_cluster)

    # The loop with everything in it
    for iteration in range(len(features) - 1):
        # Effectively doing argmin on the distance matrix
        curr_min = np.inf
        row_index = -1
        column_index = -1
        # THE STEPS HERE REFER TO BREAKING THE TIE
        # 1st step -- find the absolute minimum
        for i in range(len(distance_matrix)):
            for j in range(len(distance_matrix)):
                if distance_matrix[i, j] < curr_min:
                    curr_min = distance_matrix[i, j]
        # 2nd step -- find all the row and column indices with this absolute minimum
        distance_index_list = list()
        for i in range(len(distance_matrix)):
            for j in range(len(distance_matrix)):
                if distance_matrix[i, j] == curr_min:
                    if i < j:
                        distance_index_list.append((i, j))
        # 3rd step -- Create dictionary of distance indices to cluster indices for each unique cluster_index
        distance_index_2_cluster_index = dict()
        for distance_index_pair in distance_index_list:
            for distance_index_matrix in distance_index_pair:
                for curr_cluster in cluster_list:
                    for distance_index_cluster in curr_cluster.indices:
                        if distance_index_matrix == distance_index_cluster:
                            if curr_cluster.cluster_index not in distance_index_2_cluster_index.values():
                                distance_index_2_cluster_index[distance_index_matrix] = curr_cluster.cluster_index
        # 4th step -- Convert indices of distance matrix to cluster indices
        cluster_indices = list()
        for distance_index_pair in distance_index_list:
            if distance_index_pair[0] in distance_index_2_cluster_index.keys():
                if distance_index_pair[1] in distance_index_2_cluster_index.keys():
                    i_cluster_index = distance_index_2_cluster_index[distance_index_pair[0]]
                    j_cluster_index = distance_index_2_cluster_index[distance_index_pair[1]]
                    cluster_indices.append((i_cluster_index, j_cluster_index))
        # 5th step -- find min first cluster index, if tie, find min second cluster index
        min_combo_index = (np.inf, np.inf)
        for cluster_index in cluster_indices:
            if cluster_index[0] < min_combo_index[0]:
                min_combo_index = cluster_index
            elif cluster_index[0] == min_combo_index[0]:
                if cluster_index[1] < min_combo_index[1]:
                    min_combo_index = cluster_index
        # 6th step -- loop back through clusters to find these & assign prevcluster1 + prevcluster2
        prev_cluster1 = None
        prev_cluster2 = None
        cluster1_list_index = None
        cluster2_list_index = None
        for i_cluster in range(len(cluster_list)):
            curr_cluster = cluster_list[i_cluster]
            cluster_index = curr_cluster.cluster_index
            if cluster_index == min_combo_index[0]:
                prev_cluster1 = curr_cluster
                cluster1_list_index = i_cluster
            if cluster_index == min_combo_index[1]:
                prev_cluster2 = curr_cluster
                cluster2_list_index = i_cluster
                # cluster1_index and cluster2_index are the indices in the cluster_list that need to be removed!!
        # 7th Step -- get the original row and column indices used in the distance matrix
        for key in distance_index_2_cluster_index:
            if distance_index_2_cluster_index[key] == min_combo_index[0]:
                row_index = key
            if distance_index_2_cluster_index[key] == min_combo_index[1]:
                column_index = key

        # Getting the points from the previous clusters
        prev_points1 = prev_cluster1.points
        prev_points2 = prev_cluster2.points
        curr_points = list()
        for k in range(len(prev_points1)):
            curr_points.append(prev_points1[k])
        for k in range(len(prev_points2)):
            curr_points.append(prev_points2[k])

        # Getting the indices of the previous clusters
        prev_indices1 = prev_cluster1.indices
        prev_indices2 = prev_cluster2.indices
        curr_indices = list()
        for k in range(len(prev_indices1)):
            curr_indices.append(prev_indices1[k])
        for k in range(len(prev_indices2)):
            curr_indices.append(prev_indices2[k])

        # Getting the current linkage distance and i
        curr_linkage_distance = distance_matrix[row_index, column_index]
        curr_cluster_index = int(len(features) + iteration)

        # Putting it all into a new cluster and appending
        new_cluster = cluster(curr_indices, curr_points, curr_linkage_distance, curr_cluster_index)
        cluster_list.append(new_cluster)

        # Creating Z
        prev_cluster_index1 = int(prev_cluster1.cluster_index)
        prev_cluster_index2 = int(prev_cluster2.cluster_index)
        if prev_cluster_index1 < prev_cluster_index2:
            Z[iteration, 0] = prev_cluster_index1
            Z[iteration, 1] = prev_cluster_index2
        else:
            Z[iteration, 0] = prev_cluster_index2
            Z[iteration, 1] = prev_cluster_index1
        Z[iteration, 2] = curr_linkage_distance
        Z[iteration, 3] = len(curr_indices)

        # Taking out the old clusters
        if cluster2_list_index > cluster1_list_index:
            cluster_list.pop(cluster2_list_index)
            cluster_list.pop(cluster1_list_index)
        else:
            cluster_list.pop(cluster1_list_index)
            cluster_list.pop(cluster2_list_index)

        # Updating the distance matrix!!
        indices_to_update = new_cluster.indices
        for i in range(len(features)):
            cluster_distances = list()
            for index in indices_to_update:
                cluster_distances.append(distance_matrix[i, index])
                cluster_distances.append(distance_matrix[index, i])
            # Removing the infinities out of the cluster distances
            for i_distance in range(len(cluster_distances)):
                if cluster_distances[i_distance] == np.inf:
                    cluster_distances[i_distance] = 0
            max_distance = max(cluster_distances)
            for index in indices_to_update:
                distance_matrix[i, index] = max_distance
                distance_matrix[index, i] = max_distance
            for index1 in indices_to_update:
                for index2 in indices_to_update:
                    distance_matrix[index1, index2] = np.inf

    return Z


def imshow_hac(Z):
    dendrogram(Z)
    plt.show()


if __name__ == "__main__":
    # Loading all the data in the correct format
    dataset = load_data("Pokemon.csv")
    all_features = list()
    for i in range(len(dataset)):
        all_features.append(calc_features(dataset[i]))
    # Sending it through hac
    num_pokemon = 15
    Z = hac(all_features[:num_pokemon])
    # Plotting it
    imshow_hac(Z)

    # Testing it against dendrogram()
    # Z_test = linkage(all_features[:num_pokemon], method='complete')
    # dendrogram(Z_test)
    # plt.show()

    # Testing it against linkage()
    # num_pokemon = 20
    # print(hac([calc_features(row) for row in load_data('Pokemon.csv')][:num_pokemon]))
    # print(linkage([calc_features(row) for row in load_data('Pokemon.csv')][:num_pokemon], method='complete'))
