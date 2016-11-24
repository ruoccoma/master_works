from image_database_helper import fetch_all_image_vector_pairs
import pickle

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from helpers.image_helpers import printProgress


def kmeans_clustering(vectors, n_clusters=1000):
	print("Creating cluster...")
	cluster = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, init_size=3000)
	# cluster = DBSCAN(metric=cosine_similarity)
	cluster.fit(vectors)
	# f = open("image_vector_cluster.pickle", "wb")
	# pickle.dump(cluster, f)
	# f.close()
	return cluster


def compare_to_cluster(vector, image_cluster, k, members_dict):
	predicted_cluster_id = image_cluster.predict(vector)[0]
	members_list = members_dict[predicted_cluster_id]
	cluster_member_filenames = [x[0] for x in members_list]
	cluster_member_vectors = [x[1] for x in members_list]

	similarities = []
	# print("Cluster size: ", len(cluster_member_filenames))
	# for i in range(len(cluster_member_filenames)):
	# 	similarity = cosine_similarity(vector, [cluster_member_vectors[i]])
	# 	similarities.append((cluster_member_filenames[i], similarity))
	cosine_similarities = cosine_similarity(vector, cluster_member_vectors)[0]
	for i in range(len(cluster_member_filenames)):
		similarities.append((cluster_member_filenames[i], cosine_similarities[i]))

	similarities.sort(key=lambda s: s[1], reverse=True)
	most_similar_filenames = ["" for x in range(k)]
	k_similarities = similarities[:k]
	for index in range(len(k_similarities)):
		filname_similarity_tuple = k_similarities[index]
		most_similar_filenames[index] = filname_similarity_tuple[0]
	return most_similar_filenames, predicted_cluster_id


def get_member_ids_dict(all_image_filenames, all_image_vectors, cluster):
	member_ids_dict = dict()
	ids = cluster.labels_
	tot = len(all_image_filenames)
	for i in range(tot):
		id = ids[i]
		if id in member_ids_dict:
			member_list = member_ids_dict[id]
			member_list.append((all_image_filenames[i], all_image_vectors[i]))
			member_ids_dict[id] = member_list
		else:
			member_ids_dict[id] = [(all_image_filenames[i], all_image_vectors[i])]
		printProgress(i, tot, prefix="Create members dictionary")
	return member_ids_dict


def get_dict_cluster_sizes(cluster):
	cluster_dict = {}
	for id in cluster.labels_:
		if id in cluster_dict:
			count = cluster_dict[id]
			cluster_dict[id] = count + 1
		else:
			cluster_dict[id] = 1
	return cluster_dict


if __name__ == "__main__":
	print("Getting image vector pairs...")
	images = fetch_all_image_vector_pairs()
	filenames = [x[0] for x in images]
	image_vectors = [x[1] for x in images]
	cluster = kmeans_clustering(image_vectors, n_clusters=100)
	members_dict = get_member_ids_dict(filenames, image_vectors, cluster)
	cluster_dict = get_dict_cluster_sizes(cluster)
	max_cluster_size = 0
	print("No clusters: ", len(cluster_dict))
	for i in cluster_dict:
		# print("Cluster: ", i, " size: ", cluster_dict[i])
		if cluster_dict[i] > max_cluster_size:
			max_cluster_size = cluster_dict[i]
	print("Largest cluster: ", max_cluster_size)
	print("Query: ", filenames[0])
	most_similar = compare_to_cluster([image_vectors[0]], cluster, 5, members_dict)
	print("Most similar: ", most_similar)
