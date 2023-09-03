import os
import tensorflow as tf
import numpy as np


class Autonomous_System_Dataset():
	def __init__(self, data_dir=None):
		self.STARTING_LINE = 4
		self.SEP='\t'
		self.TYPE_FN = int
		self.STEPS_ACCOUNTED = 100
		self.data_dir = data_dir
		self.edges = self.load_edges()
		

	def load_edges(self):
		if self.data_dir == None:
			self.data_dir = 'data/AS-733/'

		files = os.listdir(self.data_dir)
		files = np.sort(files)

		edges = []

		for i in range(self.STEPS_ACCOUNTED):
			file = files[i]
			f = open(self.data_dir + file, 'r')
			lines = f.read()
			lines=lines.splitlines()
			data = [[self.TYPE_FN(r) for r in row.split(self.SEP)] for row in lines[self.STARTING_LINE:]]
			data = tf.convert_to_tensor(data)
			time_col = tf.zeros((data.shape[0],1)) + i
			time_col = tf.cast(time_col, tf.int32)
			data = tf.concat([data,time_col], axis=1)
			data = tf.concat([data, tf.gather(data, [1, 0, 2], axis=1)], axis=0)
			edges.append(data)

		edges = tf.concat(edges, axis=0)

		edges = edges.numpy()
		_, indices = np.unique(edges[:, 0:2], return_inverse=True)
		reshaped_indices = indices.reshape(edges[:, :2].shape)
		edges[:, :2] = reshaped_indices
		indices_sorted = np.lexsort((edges[:, 1], edges[:, 0], edges[:, 2]))
		edges = edges[indices_sorted]

		mask = np.append([True], np.any(edges[1:] != edges[:-1], axis=1))
		edges = edges[mask]

		self.num_nodes = np.max(indices) + 1

		ids = edges[:,0] * self.num_nodes + edges[:,1]
		self.num_non_existing = float(self.num_nodes**2 - len(tf.unique(ids)[0]))

		self.max_time = tf.math.reduce_max(edges[:,2])
		self.min_time = tf.math.reduce_min(edges[:,2])

		return {'idx': edges, 'vals': tf.ones(edges.shape[0])}

















