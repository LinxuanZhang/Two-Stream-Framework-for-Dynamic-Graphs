import os
import tensorflow as tf
import numpy as np


class UC_Irvine_Dataset():
	def __init__(self, data_dir=None):
		self.STARTING_LINE = 2
		self.SEP=' '
		self.TYPE_FN = int
		self.data_dir = data_dir
		self.aggr_time = 190080
		self.edges = self.load_edges()
		

	def load_edges(self):
		if self.data_dir == None:
			self.data_dir = 'data/UCI/'

		edges = []

		file = os.listdir(self.data_dir)[0]
		f = open(self.data_dir + file, 'r')
		lines = f.read()
		lines=lines.splitlines()
		data = [[self.TYPE_FN(r) for r in row.split(self.SEP)] for row in lines[self.STARTING_LINE:]]
		data = np.array(data)

		# aggregate by time
		data[:,3] = data[:,3] - data[:,3].min()
		data[:,3] //= self.aggr_time

		self.num_nodes = data[:, 0:2].max()
		data[:, 0:2] -= 1
		data = np.concatenate([data, data[:, [1, 0, 2, 3]]])
		

		
		ids = data[:,0] * self.num_nodes + data[:,1]

		self.num_non_existing = float(self.num_nodes**2 - len(tf.unique(ids)[0]))

		self.max_time = data[:, 3].max()
		self.min_time = data[:, 3].min()

		return {'idx': data[:, [0, 1, 3]], 'vals': tf.ones(data.shape[0])}














