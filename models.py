import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class TwoStream_GCN(keras.Model):
    def __init__(self, spatial_input_dim,
                 temporal_input_dim, 
                 spatial_hidden_size,
                 temporal_hidden_size,
                 classifier_hidden_size,
                 gcn_fusion_size, 
                 ffn_fusion_size,
                 ffn_hiden_size):
        super().__init__()
        self.spatial_gcn_1 = sota_GCN(spatial_input_dim, spatial_hidden_size)
        self.spatial_gcn_2 = sota_GCN(spatial_hidden_size, gcn_fusion_size)
        self.temporal_gcn_1 = sota_GCN(temporal_input_dim, temporal_hidden_size)
        self.temporal_gcn_2 = sota_GCN(temporal_hidden_size, gcn_fusion_size)
        
        self.temporal_classifier = keras.Sequential(
            [
                layers.Dense(classifier_hidden_size, activation='relu'),
                layers.Dense(ffn_fusion_size)
            ])
        self.spatial_classifier = keras.Sequential(
            [
                layers.Dense(classifier_hidden_size,activation='relu'),
                layers.Dense(ffn_fusion_size)
            ])
        self.classifier = keras.Sequential(
            [
                layers.Dense(ffn_fusion_size*2, activation='relu'),
                layers.Dense(ffn_hiden_size, activation='relu'),
                layers.Dense(2)
            ])

    def call(self, s, training=True):
        spatial_adj = s['hist_adj']
        spatial_feat = s['hist_ndFeats']
        temporal_adj = s['ts_adj']
        temporal_feat = s['ts_feat']

        spatial_feat = self.spatial_gcn_1(spatial_adj, spatial_feat, training)
        spatial_feat = self.spatial_gcn_2(spatial_adj, spatial_feat, training)
        
        temporal_feat = self.temporal_gcn_1(temporal_adj, temporal_feat, training)
        temporal_feat = self.temporal_gcn_2(temporal_adj, temporal_feat, training)

        pred = self.predict(spatial_feat, temporal_feat, s['label_sp'])

        return pred


    def predict(self, spatial_nodes_embs, temporal_nodes_embs, node_indices):
        predict_batch_size = 10000
        gather_predictions=[]
        node_indices = node_indices.indices
        for i in range(1 +(node_indices.shape[0]//predict_batch_size)):
            indices = np.array(range(i*predict_batch_size,(i+1)*predict_batch_size))
            indices = indices[indices<node_indices.shape[0]]

            spatial_cls_input = tf.gather(spatial_nodes_embs, tf.gather(node_indices, indices))
            spatial_cls_input = tf.reshape(spatial_cls_input, [tf.shape(spatial_cls_input)[0], -1])

            temporal_cls_input = tf.gather(temporal_nodes_embs, tf.gather(node_indices, indices))
            temporal_cls_input = tf.reshape(temporal_cls_input, [tf.shape(temporal_cls_input)[0], -1])

            spatial_predictions = self.spatial_classifier(spatial_cls_input)
            temporal_predictions = self.temporal_classifier(temporal_cls_input)

            predictions = tf.concat([spatial_predictions, temporal_predictions], axis=1)
            gather_predictions.append(predictions)
        gather_predictions = tf.concat(gather_predictions, axis=0)
        gather_predictions = self.classifier(gather_predictions)
        return gather_predictions


class Temporal_GCN(keras.Model):
    def __init__(self, 
                 temporal_input_dim, 
                 temporal_hidden_size,
                 classifier_hidden_size,
                 gcn_fusion_size):
        super().__init__()
        self.temporal_gcn_1 = sota_GCN(temporal_input_dim, temporal_hidden_size)
        self.temporal_gcn_2 = sota_GCN(temporal_hidden_size, gcn_fusion_size)
        
        self.temporal_classifier = keras.Sequential(
            [
                layers.Dense(classifier_hidden_size, activation='relu'),
                layers.Dense(2)
            ])


    def call(self, s, training=True):

        temporal_adj = s['ts_adj']
        temporal_feat = s['ts_feat']
        
        temporal_feat = self.temporal_gcn_1(temporal_adj, temporal_feat, training)
        temporal_feat = self.temporal_gcn_2(temporal_adj, temporal_feat, training)

        pred = self.predict(temporal_feat, s['label_sp'])

        return pred


    def predict(self, temporal_nodes_embs, node_indices):
        predict_batch_size = 10000
        gather_predictions=[]
        node_indices = node_indices.indices
        for i in range(1 +(node_indices.shape[0]//predict_batch_size)):
            indices = np.array(range(i*predict_batch_size,(i+1)*predict_batch_size))
            indices = indices[indices<node_indices.shape[0]]

            temporal_cls_input = tf.gather(temporal_nodes_embs, tf.gather(node_indices, indices))
            temporal_cls_input = tf.reshape(temporal_cls_input, [tf.shape(temporal_cls_input)[0], -1])
            temporal_predictions = self.temporal_classifier(temporal_cls_input)

            gather_predictions.append(temporal_predictions)

        gather_predictions = tf.concat(gather_predictions, axis=0)
        return gather_predictions


class Baseline_node2vec(keras.Model):
    def __init__(self, spatial_input_dim):
        super().__init__()       
        self.spatial_classifier = keras.Sequential(
            [
                layers.Dense(spatial_input_dim,activation='relu'),
                layers.Dense(2)
            ])


    def call(self, s):
        # spatial_feat = tf.cast(s['hist_ndFeats'], tf.float16)
        spatial_feat = s['hist_ndFeats']
        pred = self.predict(spatial_feat, s['label_sp'])
        return pred


    def predict(self, spatial_nodes_embs, node_indices):
        predict_batch_size = 10000
        gather_predictions=[]
        node_indices = node_indices.indices
        for i in range(1 +(node_indices.shape[0]//predict_batch_size)):
            indices = np.array(range(i*predict_batch_size,(i+1)*predict_batch_size))
            indices = indices[indices<node_indices.shape[0]]

            spatial_cls_input = tf.gather(spatial_nodes_embs, tf.gather(node_indices, indices))
            spatial_cls_input = tf.reshape(spatial_cls_input, [tf.shape(spatial_cls_input)[0], -1])

            predictions = self.spatial_classifier(spatial_cls_input)
            gather_predictions.append(predictions)

        gather_predictions = tf.concat(gather_predictions, axis=0)
        return gather_predictions


class Baseline_GCN(keras.Model):
    def __init__(self, spatial_input_dim, spatial_hidden_size):
        super().__init__()
        self.gcn = GCN(spatial_input_dim, spatial_hidden_size)
        self.spatial_classifier = keras.Sequential(
            [
                layers.Dense(spatial_hidden_size,activation='relu'),
                layers.Dense(2)
            ])


    def call(self, s):
        spatial_feat = s['hist_ndFeats']
        spatial_adj = s['hist_adj']
        spatial_feat = self.gcn(spatial_adj, spatial_feat)
        pred = self.predict(spatial_feat, s['label_sp'])
        return pred


    def predict(self, spatial_nodes_embs, node_indices):
        predict_batch_size = 10000
        gather_predictions=[]
        node_indices = node_indices.indices
        for i in range(1 +(node_indices.shape[0]//predict_batch_size)):
            indices = np.array(range(i*predict_batch_size,(i+1)*predict_batch_size))
            indices = indices[indices<node_indices.shape[0]]

            spatial_cls_input = tf.gather(spatial_nodes_embs, tf.gather(node_indices, indices))
            spatial_cls_input = tf.reshape(spatial_cls_input, [tf.shape(spatial_cls_input)[0], -1])

            predictions = self.spatial_classifier(spatial_cls_input)
            gather_predictions.append(predictions)

        gather_predictions = tf.concat(gather_predictions, axis=0)
        return gather_predictions


class sota_GCN(keras.Model):
    def __init__(self, in_feats, out_feats, activation=keras.activations.relu):
        super().__init__()
        self.rows = in_feats
        self.cols = out_feats
        self.activation = activation
        self.GCN_weight = tf.Variable(tf.keras.initializers.LecunNormal()(shape=(in_feats, out_feats)),
                                      name='GCN weight')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.skip_weight = tf.Variable(tf.keras.initializers.LecunNormal()(shape=(in_feats, out_feats)),
                                       name='skip weight') if in_feats != out_feats else None
        
    def call(self, Ahat, node_embs, training=True):
        if isinstance(node_embs, tf.SparseTensor):
            node_embs = tf.sparse.to_dense(node_embs)
        node_embs = tf.cast(node_embs, tf.float32)
        original_node_embs = node_embs
        node_embs = tf.sparse.sparse_dense_matmul(Ahat, node_embs)
        node_embs = tf.matmul(node_embs, self.GCN_weight)
        # batch norm
        node_embs = self.batch_norm(node_embs, training=training)
        # Skip Connection: Adjust original embeddings if dimensions are different
        if self.skip_weight is not None:
            transformed_skip = tf.matmul(original_node_embs, self.skip_weight)
            node_embs += transformed_skip
        else:
            node_embs += original_node_embs
        node_embs = self.activation(node_embs)
        return node_embs
    

class GCN(keras.Model):
    def __init__(self, in_feats, out_feats, activation=keras.activations.relu):
        super().__init__()
        self.rows = in_feats
        self.cols = out_feats
        self.activation = activation
        self.GCN_weight = tf.Variable(tf.keras.initializers.LecunNormal()(shape=(in_feats, out_feats)),
                                      name='GCN weight')
        
    def call(self, Ahat, node_embs):
        if isinstance(node_embs, tf.SparseTensor):
            node_embs = tf.sparse.to_dense(node_embs)
        node_embs = tf.sparse.sparse_dense_matmul(Ahat, node_embs)
        node_embs = tf.matmul(node_embs, self.GCN_weight)
        node_embs = self.activation(node_embs)
        return node_embs
    





