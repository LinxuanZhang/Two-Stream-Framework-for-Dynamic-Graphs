import os
import tensorflow as tf
import numpy as np


class Bitcoin_Dataset():
	def __init__(self, data_dir=None):
		self.data_dir = data_dir
		
		self.aggr_time = 1200000
		self.edges = self.load_edges()
		

	def load_edges(self):
		assert self.data_dir is not None

		# load file
		file = os.listdir(self.data_dir)[0]
		f = open(self.data_dir + file, 'r')
		lines = f.read().splitlines()
		data = [[float(r) for r in row.split(',')] for row in lines]
		edges = np.array(data).astype(np.int64)

		# relabel indices
		_, indices = np.unique(edges[:, 0:2], return_inverse=True)
		reshaped_indices = indices.reshape(edges[:, :2].shape)
		edges[:, :2] = reshaped_indices

		self.num_nodes = edges[:, :2].max() + 1

		# normalize time col
		edges[:, 3] = edges[:, 3] - edges[:, 3].min()
		edges[:,3] = edges[:,3]//self.aggr_time

		# identify positive/negative link
		pos_indices = edges[:, 2]>0
		edges[pos_indices, 2] = 1
		edges[~pos_indices, 2] = -1
				
		# add reversed link to make graph undirected
		edges = np.concatenate([edges, edges[:, [1, 0, 2, 3]]], axis=0)

		self.max_time = edges[:, 3].max()
		self.min_time = edges[:, 3].min()
		
		# Below code for edge classification

		# separate class
		sp_indices = edges[:, [0, 1, 3]]
		sp_values = edges[:, 2]

		neg_mask = sp_values == -1
		neg_sp_indices = sp_indices[neg_mask,:]
		neg_sp_values = sp_values[neg_mask]

		neg_sp_indices, neg_sp_values = self.coalesce(neg_sp_indices, neg_sp_values)

		neg_sp_edges = tf.sparse.SparseTensor(indices=neg_sp_indices, values=neg_sp_values, 
                                       dense_shape=[self.num_nodes, self.num_nodes, self.max_time+1])


		pos_mask = sp_values == 1
		pos_sp_indices = sp_indices[pos_mask,:]
		pos_sp_values = sp_values[pos_mask]
		pos_sp_indices, pos_sp_values = self.coalesce(pos_sp_indices, pos_sp_values)
		pos_sp_edges = tf.sparse.SparseTensor(indices=pos_sp_indices, values=pos_sp_values, 
                                       dense_shape=[self.num_nodes, self.num_nodes, self.max_time+1])
		
		#scale positive class to separate after adding
		pos_sp_edges *= 1000

		#we substract the neg_sp_edges to make the values positive
		sp_edges = tf.sparse.add(pos_sp_edges, neg_sp_edges*-1)
		sp_i, sp_v = self.coalesce(sp_edges.indices.numpy(), sp_edges.values.numpy())
		sp_edges = tf.sparse.SparseTensor(indices=sp_i, values=sp_v,
										dense_shape=[self.num_nodes, self.num_nodes, self.max_time+1])

		
        #separating negs and positive edges per edge/timestamp
		vals = sp_edges.values
		neg_vals = vals.numpy()%1000
		pos_vals = vals.numpy()//1000		

        # #We add the negative and positive scores and do majority voting
		vals = pos_vals - neg_vals

        #creating labels new_vals -> the label of the edges
		new_vals = np.zeros(vals.shape[0])
		new_vals[vals>0] = 1
		new_vals[vals<=0] = 0
		indices_labels = np.concatenate([sp_edges.indices.numpy(), new_vals.reshape(-1, 1)], axis=1).astype(int)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
		vals = pos_vals + neg_vals
		vals = vals.astype(np.float32)
		return {'idx': indices_labels, 'vals': vals}

	def coalesce(self, indices, values):
		aggregated_values = {}
		edges = np.concatenate([indices, values.reshape(-1, 1)], axis=1)
		# Iterate through the rows of the input array
		for row in edges:
			from_node, to_node, time, value = row
			# Create a tuple (from_node, to_node, time) as the key for the dictionary
			key = (from_node, to_node, time)
			# If the key exists in the dictionary, add the value to the existing total
			if key in aggregated_values:
				aggregated_values[key] += value
			else:
				# If the key does not exist, create a new entry with the value
				aggregated_values[key] = value
		# Convert the dictionary values to a numpy array
		edges = np.array([[k[0], k[1], v, k[2]] for k, v in aggregated_values.items()])
		indices = edges[:, [0, 1, 3]]
		values = edges[:, 2]
		return indices, values

















