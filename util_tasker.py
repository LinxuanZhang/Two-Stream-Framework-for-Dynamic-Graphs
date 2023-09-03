import numpy as np
import tensorflow as tf
import time
import random
import pickle
from sklearn.decomposition import TruncatedSVD

def set_seeds(rank):
    seed = int(time.time())+rank
    np.random.seed(seed)
    random.seed(seed)


class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_max_degs(dataset,all_window=False):
    max_deg_out = []
    max_deg_in = []
    for t in range(dataset.min_time, dataset.max_time):
        if all_window:
            window = t+1
        else:
            window = 1
        cur_adj = get_sp_adj(data = dataset,
                             time = t,
                             weighted = False,
                             time_window = window)
        # print(window)
        cur_out, cur_in = get_degree_vects(cur_adj,dataset.num_nodes)
        max_deg_out.append(np.nanmax(cur_out.numpy()))
        max_deg_in.append(np.nanmax(cur_in.numpy()))    
    max_deg_out = np.nanmax(max_deg_out)
    max_deg_in = np.nanmax(max_deg_in)
    max_deg_out = int(max_deg_out) + 1
    max_deg_in = int(max_deg_in) + 1
    return max_deg_out, max_deg_in


def get_degree_vects(adj,num_nodes):
    adj = tf.sparse.SparseTensor(tf.cast(adj['idx'], tf.int64),adj['vals'],[num_nodes, num_nodes])
    degs_out = tf.sparse.sparse_dense_matmul(adj, tf.ones((num_nodes, 1)))
    degs_in = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(adj), 
                    tf.ones((num_nodes,1)))
    return degs_out, degs_in


def normalize_adj(adj,num_nodes):
    '''
    takes an adj matrix as a dict with idx and vals and normalize it by: 
        - adding an identity matrix, 
        - computing the degree vector
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2
    '''
    sp_tensor = tf.sparse.SparseTensor(tf.cast(adj['idx'], tf.int64),adj['vals'],[num_nodes, num_nodes])
    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = tf.sparse.add(sparse_eye, sp_tensor)
    idx = sp_tensor.indices
    vals = sp_tensor.values
    degree = tf.sparse.reduce_sum(sp_tensor, axis=1)
    di = tf.gather(degree, idx[:, 0])
    dj = tf.gather(degree, idx[:, 1])
    vals = vals * ((di * dj) ** -0.5)
    return tf.sparse.SparseTensor(tf.cast(idx, tf.int64),vals,[num_nodes, num_nodes])

def make_sparse_eye(size):
    eye_idx = tf.range(size)
    eye_idx = tf.stack([eye_idx,eye_idx],axis=1)
    vals = tf.ones(size)
    eye = tf.sparse.SparseTensor(tf.cast(eye_idx, tf.int64),vals,[size,size])
    return eye


def get_1_hot_deg_feats(adj,max_deg,num_nodes):
    new_vals = tf.ones(adj['idx'].shape[0])
    new_adj = {'idx':adj['idx'], 'vals': new_vals}
    degs_out, _ = get_degree_vects(new_adj,num_nodes)
    degs_out = {'idx': tf.concat([tf.reshape(tf.range(num_nodes),[-1,1]),
                                  tf.cast(tf.reshape(degs_out,[-1,1]), tf.int64)],
                                  axis=1),
                'vals': tf.ones(num_nodes)}
    degs_out = tf.sparse.SparseTensor(tf.cast(degs_out['idx'], tf.int64),degs_out['vals'],[num_nodes, max_deg])

    return degs_out



def get_sp_adj(data,time,weighted=False,time_window = 1):
    edges = data.edges
    idx = edges['idx']
    subset = (idx[:, 2] <= time)&(idx[:, 2]>(time-time_window))
    idx = edges['idx'][subset][:,0:2]
    vals = edges['vals'][subset]
    if not weighted:
        vals = tf.ones(idx.shape[0])
    return {'idx': idx, 'vals': vals}


def get_node_mask(cur_adj,num_nodes):
    mask = tf.zeros(num_nodes) - float("Inf")
    non_zero = np.unique(cur_adj['idx'])
    mask = tf.tensor_scatter_nd_update(tensor=mask, indices=non_zero.reshape(-1, 1), updates=tf.zeros(len(non_zero)))
    return mask


def get_edges_ids(sp_idx, tot_nodes):

    return sp_idx[:,0]*tot_nodes + sp_idx[:,1]


def sample_new_edges(adj, existing_nodes, num_new_edges_per_node=1000):
    # Convert to set for O(1) lookup times. Account for both edge orientations since the graph is undirected.
    edges_set = set(map(tuple, adj))
    edges_set = edges_set.union({(b, a) for a, b in edges_set})

    from_nodes = np.unique(adj[:, 0])
    sampled_edges = []

    for from_node in from_nodes:
        new_edges_for_node = []
        potential_to_nodes = np.array(list(set(existing_nodes) - {from_node}))
        np.random.shuffle(potential_to_nodes)
        
        for to_node in potential_to_nodes:
            if (from_node, to_node) not in edges_set:
                new_edges_for_node.append([from_node, to_node])
                if len(new_edges_for_node) == num_new_edges_per_node:
                    break
        
        sampled_edges.extend(new_edges_for_node)

    sampled_edges = np.array(sampled_edges)
    
    new_adj = {'idx': sampled_edges, 'vals': tf.cast(tf.zeros(sampled_edges.shape[0]), tf.float32)}
    return new_adj


def get_non_existing_edges(adj,number, tot_nodes, smart_sampling, existing_nodes=None):
    idx = adj['idx']
    true_ids = get_edges_ids(idx,tot_nodes)
    true_ids = set(true_ids)
    #the maximum of edges would be all edges that don't exist between nodes that have edges
    num_edges = min(number,idx.shape[0] * (idx.shape[0]-1) - len(true_ids))
    if smart_sampling:
        def sample_edges(num_edges):
            from_id = np.random.choice(idx[:, 0],size = num_edges,replace = True)
            to_id = np.random.choice(existing_nodes,size = num_edges, replace = True)
            if num_edges>1:
                edges = np.stack([from_id,to_id])
            else:
                edges = np.concatenate([from_id,to_id])
            return edges
    else:
        def sample_edges(num_edges):
            if num_edges > 1:
                edges = np.random.randint(0,tot_nodes,(2,num_edges))
            else:
                edges = np.random.randint(0,tot_nodes,(2,))
            return edges
    edges = sample_edges(num_edges*4)
    edge_ids = edges[0] * tot_nodes + edges[1]
    out_ids = set()
    num_sampled = 0
    sampled_indices = []
    for i in range(num_edges*4):
        eid = edge_ids[i]
        #ignore if any of these conditions happen
        if eid in out_ids or edges[0,i] == edges[1,i] or eid in true_ids:
            continue
        #add the eid and the index to a list
        out_ids.add(eid)
        sampled_indices.append(i)
        num_sampled += 1
        #if we have sampled enough edges break
        if num_sampled >= num_edges:
            break
    edges = edges[:,sampled_indices]
    edges = tf.transpose(tf.convert_to_tensor(edges))
    vals = tf.zeros(edges.shape[0])
    return {'idx': edges, 'vals': vals}


def get_all_non_existing_edges(adj,tot_nodes):
    # true_ids = adj['idx'].numpy()
    true_ids = adj['idx']
    true_ids = get_edges_ids(true_ids,tot_nodes)
    all_edges_idx = np.arange(tot_nodes)
    all_edges_idx = np.array(np.meshgrid(all_edges_idx, all_edges_idx)).reshape(2,-1).T
    all_edges_ids = get_edges_ids(all_edges_idx,tot_nodes)
    #only edges that are not in the true_ids should keep here
    mask = np.logical_not(np.isin(all_edges_ids,true_ids))
    non_existing_edges_idx = all_edges_idx[mask]
    edges = tf.convert_to_tensor(non_existing_edges_idx)
    vals = tf.zeros(edges.shape[0])
    return {'idx': edges, 'vals': vals}


def save_pkl(obj, path):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)


def load_pkl(path):
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def cal_avg_graph(cur_hist, major_threshold=None):
    cur_hist_list = [c['idx'] for c in cur_hist]
    cur_hist_list = tf.concat(cur_hist_list, axis=0)
    tf.raw_ops.UniqueV2(x=cur_hist_list, axis=[0])
    edges, counts = np.unique(cur_hist_list, axis=0, return_counts=True)
    if major_threshold is None:
        major_threshold = len(cur_hist)//2+1
    edges = edges[counts>major_threshold]
    return {'idx':edges, 'vals':tf.ones(edges.shape[0])}


def merge_deg(cur_hist_features):
    merged_deg = np.stack([c.indices[:,1] for c in cur_hist_features]).T
    merged_deg = tf.cast(merged_deg, tf.float32)
    return merged_deg


from gensim.models import Word2Vec

def edges_to_adj_list(edges):
    graph = {}
    for edge in edges:
        if edge[0] not in graph:
            graph[edge[0]] = []
        if edge[1] not in graph:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])  # Assuming undirected graph
    return graph


def random_walk(graph, start_node, walk_length=10, p=1, q=1):
    """
    Generate a single random walk from a starting node.

    Parameters:
        graph: dict
            A dictionary where keys are nodes and values are lists of neighboring nodes.
        start_node: any hashable
            The starting node of the walk.
        walk_length: int
            Length of the random walk.
        p: float
            Return parameter (controls likelihood of revisiting a node already visited in the walk).
        q: float
            In-out parameter (controls likelihood of visiting nodes not directly connected to the current node).

    Returns:
        list
            A list of nodes representing the random walk.
    """
    walk = [start_node]
    current_node = start_node

    for _ in range(walk_length - 1):
        neighbors = graph[current_node]
        if len(neighbors) == 0:
            break

        if len(walk) == 1:
            next_node = random.choice(neighbors)
        else:
            prev_node = walk[-2]
            probs = []
            for neighbor in neighbors:
                if neighbor == prev_node:
                    probs.append(1/p)
                elif neighbor in graph[prev_node]:
                    probs.append(1)
                else:
                    probs.append(1/q)
            prob_sum = sum(probs)
            probs = [p/prob_sum for p in probs]
            next_node = random.choices(neighbors, weights=probs)[0]

        walk.append(next_node)
        current_node = next_node

    return walk


def node2vec(edges, dimensions=128, walk_length=40, num_walks=10, p=1, q=1, window=5, min_count=1, workers=4):
    """
    Generate node embeddings using Node2Vec.

    Parameters:
        edges: list of list
            List of edges where each edge is a list of two nodes.
        dimensions: int
            Dimensionality of the embeddings.
        walk_length: int
            Length of the random walk.
        num_walks: int
            Number of walks per node.
        p: float
            Return parameter.
        q: float
            In-out parameter.
        window: int
            Window size for the skip-gram model.
        min_count: int
            Ignores all words with total frequency lower than this in the skip-gram model.
        workers: int
            Number of CPU cores to use in the training.

    Returns:
        gensim.models.word2vec.Word2Vec
            The trained model.
    """
    
    # Convert edges to adjacency list
    graph = edges_to_adj_list(edges)

    walks = []
    for node in graph:
        for _ in range(num_walks):
            walk = random_walk(graph, node, walk_length, p, q)
            walks.append([str(node) for node in walk])

    model = Word2Vec(sentences=walks, vector_size=dimensions, window=window, min_count=min_count, workers=workers)
    return model


def sparse_to_dense(sparse_tensor):

    return tf.sparse.to_dense(sparse_tensor)


def reduce_dimension(sparse_tensor, embs_dim):
    # Convert sparse tensor to dense tensor
    dense_tensor = sparse_to_dense(sparse_tensor)
    # Convert TensorFlow tensor to numpy array
    numpy_array = dense_tensor.numpy()
    # Apply truncated SVD
    svd = TruncatedSVD(n_components=embs_dim)
    reduced_data = svd.fit_transform(numpy_array)
    # Convert numpy array back to TensorFlow tensor
    reduced_tensor = tf.convert_to_tensor(reduced_data)
    return reduced_tensor