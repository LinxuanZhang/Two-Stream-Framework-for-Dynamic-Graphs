import tensorflow as tf
import numpy as np
import os
import data_auto_sys as aus
import util_tasker as ut


class Link_Pred_Tasker():
	def __init__(self, dataset, path='prep_data/AS733/', prep=False, embs_dim = 128, temp_dim = 20, 
		  walk_length=40, num_walks=10, p=1, q=1, window=5, major_threshold=None, smart_neg_sampling = False, neg_sample=1000):
		self.data = dataset
		## These could be input args in the future
		self.smart_neg_sampling = smart_neg_sampling
		self.num_hist_steps = 10
		self.neg_mult = 100
		self.adj_mat_time_window = 1
		## These will be the attribute specific to Link_Pred_Tasker
		self.num_class = 2
		self.is_static = False
		#max_time for link pred should be one before
		self.max_time = dataset.max_time
		self.path = path
		self.hist_loaded = False
		self.embs_dim = embs_dim
		self.temp_dim = temp_dim
		self.walk_length = walk_length
		self.num_walks = num_walks
		self.p = p
		self.q = q
		self.window=window
		self.major_threshold=major_threshold
		self.neg_sample = neg_sample
		if prep:
			self.prep_hist_sample(embs_dim=self.temp_dim)
			self.prep_hist_sample(embs_dim=self.embs_dim)
			self.prep_temporal_sample(embs_dim=self.temp_dim)
			self.prep_label_sample()
			self.load_hist_data()
		try:
			self.load_hist_data()
		except:
			print('dataset not preped!')


	def prep_hist_sample(self, embs_dim = None):
		if not embs_dim:
			embs_dim = self.embs_dim

		if not os.path.exists(self.path):
			os.makedirs(self.path)

		feat_path = self.path + f'feat_{embs_dim}/'
		if not os.path.exists(feat_path):
			os.mkdir(feat_path)

		hist_adj_list = []
		norm_hist_adj_list = []
		hist_ndFeats_list = []
		existing_nodes = []

		for i in range(self.data.min_time, self.data.max_time+1):
			print(f'preping hist graph {i} of embs_dim {embs_dim}')
			cur_adj = ut.get_sp_adj(data = self.data, 
									time = i,
									weighted = True,
									time_window = self.adj_mat_time_window)
			if self.smart_neg_sampling:
				existing_nodes.append(np.unique(cur_adj['idx']))
			else:
				existing_nodes = None
			node2vec_model = ut.node2vec(cur_adj['idx'], dimensions=embs_dim, walk_length=self.walk_length, num_walks=self.num_walks, p=self.p, q=self.q, window=self.window)
			node_feats = []
			for node in range(self.data.num_nodes):
				if str(node) in node2vec_model.wv.index_to_key:
					node_feats.append(node2vec_model.wv[str(node)])
				else:
					node_feats.append([0]*embs_dim)

			node_feats = np.array(node_feats)
			ut.save_pkl(node_feats, feat_path + f'node2vec_feat_{i}')

			hist_adj_list.append(cur_adj)
			norm_cur_adj = ut.normalize_adj(adj = cur_adj, num_nodes = self.data.num_nodes)
			norm_hist_adj_list.append(norm_cur_adj)
			hist_ndFeats_list.append(node_feats)


		ut.save_pkl(hist_adj_list, self.path + 'hist_adj_list.pkl')
		ut.save_pkl(hist_ndFeats_list, self.path + f'hist_ndFeats_list_{embs_dim}.pkl')
		ut.save_pkl(norm_hist_adj_list, self.path + 'norm_hist_adj_list.pkl')
		ut.save_pkl(existing_nodes, self.path + 'existing_nodes')


	def prep_temporal_sample(self, embs_dim=None):	
		print(f'preping temporal sample with major_threshold of {self.major_threshold}')	
		if not embs_dim:
			embs_dim=self.temp_dim
			
		self.hist_adj_list = ut.load_pkl(self.path + 'hist_adj_list.pkl')
		self.hist_ndFeats_list = ut.load_pkl(self.path + f'hist_ndFeats_list_{embs_dim}.pkl')
		temporal_stream = []
		temporal_stream_feat = []
		for i in range(self.num_hist_steps, self.data.max_time+1):
			cur_hist = self.hist_adj_list[(i-self.num_hist_steps):i]
			cur_hist_features = self.hist_ndFeats_list[(i-self.num_hist_steps):i]
			cur_ts = ut.cal_avg_graph(cur_hist, major_threshold=self.major_threshold)
			cur_ts = ut.normalize_adj(adj = cur_ts, num_nodes = self.data.num_nodes)
			cur_ts_feat = tf.concat(cur_hist_features, axis=1)
			temporal_stream.append(cur_ts)
			temporal_stream_feat.append(cur_ts_feat)

		ut.save_pkl(temporal_stream, self.path + 'temporal_stream.pkl')
		ut.save_pkl(temporal_stream_feat, self.path + 'temporal_stream_feat.pkl')


	def prep_label_sample(self):
		smart_sample=self.smart_neg_sampling
		label_path = self.path + 'label/'
		if not os.path.exists(label_path):
			os.mkdir(label_path)

		self.load_hist_data()
		ut.set_seeds(123)
		for i in range(self.num_hist_steps, self.data.max_time+1):

			label_adj = ut.get_sp_adj(data = self.data, 
						time = i,
						weighted = False,
						time_window =  self.adj_mat_time_window)

			if smart_sample:
				try:
					cur_existing_nodes = tf.concat(self.existing_nodes[(i-self.num_hist_steps):i], axis=0)
					non_exisiting_adj = ut.sample_new_edges(adj = label_adj['idx'], existing_nodes = cur_existing_nodes.numpy(), num_new_edges_per_node=self.neg_sample)
					all_edge_label_adj = label_adj
					all_edge_label_adj['idx'] = tf.concat([tf.cast(all_edge_label_adj['idx'], tf.int64),non_exisiting_adj['idx']], axis=0)
					all_edge_label_adj['vals'] = tf.concat([all_edge_label_adj['vals'],non_exisiting_adj['vals']], axis=0)
					all_edge_label_adj['idx'] = all_edge_label_adj['idx'].numpy().astype(np.int32)
					all_edge_label_adj['vals'] = all_edge_label_adj['vals'].numpy().astype(np.int32)
					ut.save_pkl(all_edge_label_adj, label_path + f'smart_sample_edge_label_{i}')
					print(f'generated label {i}')
				except:
					print('smart_sample_label_adj fail')
					print(i)
			else:
				try:
					non_exisiting_adj = ut.get_all_non_existing_edges(adj = label_adj, tot_nodes = self.data.num_nodes)
					all_edge_label_adj = label_adj
					all_edge_label_adj['idx'] = tf.concat([tf.cast(all_edge_label_adj['idx'], tf.int64),non_exisiting_adj['idx']], axis=0)
					all_edge_label_adj['vals'] = tf.concat([all_edge_label_adj['vals'],non_exisiting_adj['vals']], axis=0)
					all_edge_label_adj['idx'] = all_edge_label_adj['idx'].numpy().astype(np.int32)
					all_edge_label_adj['vals'] = all_edge_label_adj['vals'].numpy().astype(np.int32)
					ut.save_pkl(all_edge_label_adj, label_path + f'all_edge_label_{i}')
					print(f'generated label {i}')
				except:
					print('non_existing_adj fail')
					print(i)
			
	
	def load_hist_data(self):
		self.hist_loaded = True
		self.hist_adj_list = ut.load_pkl(self.path + 'hist_adj_list.pkl')
		self.hist_ndFeats_list = ut.load_pkl(self.path + f'hist_ndFeats_list_{self.embs_dim}.pkl')
		self.norm_hist_adj_list = ut.load_pkl(self.path + 'norm_hist_adj_list.pkl')
		self.temporal_stream = ut.load_pkl(self.path + 'temporal_stream.pkl')
		self.temporal_stream_feat = ut.load_pkl(self.path + 'temporal_stream_feat.pkl')
		self.existing_nodes = ut.load_pkl(self.path + 'existing_nodes')



	def load_label(self):
		self.labels = {}
		for idx in range(self.num_hist_steps, self.data.max_time+1):
			self.labels[idx] = ut.load_pkl(self.path + f'label/all_edge_label_{idx}')


	def get_sample(self, idx):
		assert idx >= self.num_hist_steps
		if not self.hist_loaded:
			self.load_hist_data()

		if self.smart_neg_sampling:
			label_adj = ut.load_pkl(self.path + f'label/smart_sample_edge_label_{idx}')
		else:
			label_adj = ut.load_pkl(self.path + f'label/all_edge_label_{idx}')
		label_adj = tf.sparse.SparseTensor(tf.cast(label_adj['idx'], tf.int64),tf.cast(label_adj['vals'], tf.float16),[self.data.num_nodes, self.data.num_nodes])
		temporal_adj_sample = self.temporal_stream[idx-self.num_hist_steps]
		temporal_adj_feat_sample = self.temporal_stream_feat[idx-self.num_hist_steps]

		hist_adj_list_sample = self.norm_hist_adj_list[idx-1]
		hist_ndFeats_list_sample = self.hist_ndFeats_list[idx-1]

		return {'idx': idx,
				'hist_adj': hist_adj_list_sample, # SparseTensor
				'hist_ndFeats': hist_ndFeats_list_sample, # SparseTensor
				'ts_adj':temporal_adj_sample, # SparseTensor
				'ts_feat':temporal_adj_feat_sample, # ndarray
				'label_sp': label_adj # SparseTensor
				}


class Link_Pred_Tasker_outdeg():
	'''
	This is the almost the same as the Link_Pred_Tasker, excpet this uses 1 hot out degree as initial node embedding
	where Link_Pred_Tasker uses Node2Vec
	'''
	def __init__(self, dataset, path='prep_data/AS733/', prep=False, embs_dim = 128, temp_dim = 20, 
		  walk_length=40, num_walks=10, p=1, q=1, window=5, major_threshold=None, smart_neg_sampling = False, neg_sample=1000):
		self.data = dataset
		## These could be input args in the future
		self.smart_neg_sampling = smart_neg_sampling
		self.num_hist_steps = 10
		self.neg_mult = 100
		self.adj_mat_time_window = 1
		## These will be the attribute specific to Link_Pred_Tasker
		self.num_class = 2
		self.is_static = False
		self.embs_dim = embs_dim
		self.temp_dim = temp_dim
		#max_time for link pred should be one before
		self.max_time = dataset.max_time
		self.path = path
		self.hist_loaded = False
		self.max_deg, _ = ut.get_max_degs(self.data)
		self.walk_length = walk_length
		self.num_walks = num_walks
		self.p = p
		self.q = q
		self.window=window
		self.major_threshold=major_threshold
		self.neg_sample = neg_sample
		if prep:
			self.prep_hist_sample()
			self.prep_temporal_sample()
			self.prep_label_sample()
			self.load_hist_data()
		try:
			self.load_hist_data()
		except:
			print('dataset not preped!')


	def prep_hist_sample(self):

		if not os.path.exists(self.path):
			os.makedirs(self.path)

		hist_adj_list = []
		norm_hist_adj_list = []
		hist_ndFeats_list = []

		existing_nodes = []
		original_ndFeats_list = []
		max_retries = 10
		for i in range(self.data.min_time, self.data.max_time+1):
			retries = 0
			successful = False
			
			while retries < max_retries and not successful:
				try:
					cur_adj = ut.get_sp_adj(data = self.data, 
											time = i,
											weighted = True,
											time_window = self.adj_mat_time_window)
					if self.smart_neg_sampling:
						existing_nodes.append(np.unique(cur_adj['idx']))
					else:
						existing_nodes = None
					node_feats = ut.get_1_hot_deg_feats(cur_adj, self.max_deg, self.data.num_nodes)
					original_ndFeats_list.append(node_feats)
					reduced_node_feats = ut.reduce_dimension(node_feats, self.embs_dim)
					hist_adj_list.append(cur_adj)
					norm_cur_adj = ut.normalize_adj(adj = cur_adj, num_nodes = self.data.num_nodes)
					norm_hist_adj_list.append(norm_cur_adj)
					hist_ndFeats_list.append(reduced_node_feats)
					successful = True
				except ValueError as e:
					if "array must not contain infs or NaNs" in str(e):
						retries += 1
						print(f"Error at iteration {i} (Retry {retries}/{max_retries}): {e}")
					else:
						raise
			if not successful:
				print(f"Failed to process data at iteration {i} after {max_retries} retries.")
				break

		ut.save_pkl(hist_adj_list, self.path + 'hist_adj_list.pkl')
		ut.save_pkl(hist_ndFeats_list, self.path + f'hist_ndFeats_list.pkl')
		ut.save_pkl(norm_hist_adj_list, self.path + 'norm_hist_adj_list.pkl')
		ut.save_pkl(original_ndFeats_list, self.path + 'original_ndFeats_list.pkl')
		ut.save_pkl(existing_nodes, self.path + 'existing_nodes')


	def prep_temporal_sample(self):	
		print(f'preping temporal sample with major_threshold of {self.major_threshold}')	
			
		self.hist_adj_list = ut.load_pkl(self.path + 'hist_adj_list.pkl')
		self.original_ndFeats_list = ut.load_pkl(self.path + f'original_ndFeats_list.pkl')
		temporal_stream = []
		temporal_stream_feat = []
		max_retries = 10
		for i in range(self.num_hist_steps, self.data.max_time+1):
			print(f'preparing temporal feature {i}')
			retries = 0
			successful = False
			
			while retries < max_retries and not successful:
				try:
					cur_hist = self.hist_adj_list[(i-self.num_hist_steps):i]
					cur_hist_features = self.original_ndFeats_list[(i-self.num_hist_steps):i]
					cur_ts = ut.cal_avg_graph(cur_hist, major_threshold=self.major_threshold)
					cur_ts = ut.normalize_adj(adj = cur_ts, num_nodes = self.data.num_nodes)

					cur_ts_feat = [ut.reduce_dimension(feats, self.temp_dim) for feats in cur_hist_features]
					cur_ts_feat = tf.concat(cur_ts_feat, axis=1)
					temporal_stream.append(cur_ts)
					temporal_stream_feat.append(cur_ts_feat)
					successful = True
				except ValueError as e:
					if "array must not contain infs or NaNs" in str(e):
						retries += 1
						print(f"Error at iteration {i} (Retry {retries}/{max_retries}): {e}")
					else:
						raise
			if not successful:
				print(f"Failed to process data at iteration {i} after {max_retries} retries.")
				break

		ut.save_pkl(temporal_stream, self.path + 'temporal_stream.pkl')
		ut.save_pkl(temporal_stream_feat, self.path + 'temporal_stream_feat.pkl')


	def prep_label_sample(self):
		smart_sample=self.smart_neg_sampling
		label_path = self.path + 'label/'
		if not os.path.exists(label_path):
			os.mkdir(label_path)

		self.load_hist_data()
		ut.set_seeds(123)
		for i in range(self.num_hist_steps, self.data.max_time+1):

			label_adj = ut.get_sp_adj(data = self.data, 
						time = i,
						weighted = False,
						time_window =  self.adj_mat_time_window)

			if smart_sample:
				try:
					cur_existing_nodes = tf.concat(self.existing_nodes[(i-self.num_hist_steps):i], axis=0)
					non_exisiting_adj = ut.sample_new_edges(adj = label_adj['idx'], existing_nodes = cur_existing_nodes.numpy(), num_new_edges_per_node=self.neg_sample)
					all_edge_label_adj = label_adj
					all_edge_label_adj['idx'] = tf.concat([tf.cast(all_edge_label_adj['idx'], tf.int64),non_exisiting_adj['idx']], axis=0)
					all_edge_label_adj['vals'] = tf.concat([all_edge_label_adj['vals'],non_exisiting_adj['vals']], axis=0)
					all_edge_label_adj['idx'] = all_edge_label_adj['idx'].numpy().astype(np.int32)
					all_edge_label_adj['vals'] = all_edge_label_adj['vals'].numpy().astype(np.int32)
					ut.save_pkl(all_edge_label_adj, label_path + f'smart_sample_edge_label_{i}')
					print(f'generated label {i}')
				except:
					print('smart_sample_label_adj fail')
					print(i)
			else:
				try:
					non_exisiting_adj = ut.get_all_non_existing_edges(adj = label_adj, tot_nodes = self.data.num_nodes)
					all_edge_label_adj = label_adj
					all_edge_label_adj['idx'] = tf.concat([tf.cast(all_edge_label_adj['idx'], tf.int64),non_exisiting_adj['idx']], axis=0)
					all_edge_label_adj['vals'] = tf.concat([all_edge_label_adj['vals'],non_exisiting_adj['vals']], axis=0)
					all_edge_label_adj['idx'] = all_edge_label_adj['idx'].numpy().astype(np.int32)
					all_edge_label_adj['vals'] = all_edge_label_adj['vals'].numpy().astype(np.int32)
					ut.save_pkl(all_edge_label_adj, label_path + f'all_edge_label_{i}')
					print(f'generated label {i}')
				except:
					print('non_existing_adj fail')
					print(i)
			
	
	def load_hist_data(self):
		self.hist_loaded = True
		self.hist_adj_list = ut.load_pkl(self.path + 'hist_adj_list.pkl')
		self.hist_ndFeats_list = ut.load_pkl(self.path + f'hist_ndFeats_list.pkl')
		self.norm_hist_adj_list = ut.load_pkl(self.path + 'norm_hist_adj_list.pkl')
		self.temporal_stream = ut.load_pkl(self.path + 'temporal_stream.pkl')
		self.temporal_stream_feat = ut.load_pkl(self.path + 'temporal_stream_feat.pkl')
		self.existing_nodes = ut.load_pkl(self.path + 'existing_nodes')


	def load_label(self):
		self.labels = {}
		for idx in range(self.num_hist_steps, self.data.max_time+1):
			self.labels[idx] = ut.load_pkl(self.path + f'label/all_edge_label_{idx}')


	def get_sample(self, idx):
		assert idx >= self.num_hist_steps
		if not self.hist_loaded:
			self.load_hist_data()

		if self.smart_neg_sampling:
			label_adj = ut.load_pkl(self.path + f'label/smart_sample_edge_label_{idx}')
		else:
			label_adj = ut.load_pkl(self.path + f'label/all_edge_label_{idx}')
		label_adj = tf.sparse.SparseTensor(tf.cast(label_adj['idx'], tf.int64),tf.cast(label_adj['vals'], tf.float16),[self.data.num_nodes, self.data.num_nodes])
		temporal_adj_sample = self.temporal_stream[idx-self.num_hist_steps]
		temporal_adj_feat_sample = self.temporal_stream_feat[idx-self.num_hist_steps]

		hist_adj_list_sample = self.norm_hist_adj_list[idx-1]
		hist_ndFeats_list_sample = self.hist_ndFeats_list[idx-1]

		return {'idx': idx,
				'hist_adj': hist_adj_list_sample, # SparseTensor
				'hist_ndFeats': hist_ndFeats_list_sample, # SparseTensor
				'ts_adj':temporal_adj_sample, # SparseTensor
				'ts_feat':temporal_adj_feat_sample, # ndarray
				'label_sp': label_adj # SparseTensor
				}
