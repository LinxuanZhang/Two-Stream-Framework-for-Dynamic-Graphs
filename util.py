import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.sparse import coo_matrix
from scipy.special import softmax


def gather_node_embs(nodes_embs,node_indices):
    cls_input = []
    for node_set in node_indices:
        cls_input.append(tf.gather(nodes_embs, node_set))
    return tf.concat(cls_input,axis=1)


def get_MRR(labels,logits, adj):
    predictions = logits.numpy()
    predictions = softmax(predictions, axis=1)
    true_classes = labels.numpy()
    adj = adj.numpy()
    pred_matrix = coo_matrix((predictions[:,1],(adj[:, 0],adj[:, 1]))).toarray()
    true_matrix = coo_matrix((true_classes.astype(np.float32),(adj[:,0],adj[:,1]))).toarray()
    row_MRRs = []
    for i,pred_row in enumerate(pred_matrix):
        #check if there are any existing edges
        if np.isin(1,true_matrix[i]):
            row_MRRs.append(get_row_MRR(pred_row,true_matrix[i]))
    avg_MRR = np.array(row_MRRs).mean()
    return avg_MRR


def get_row_MRR(probs,true_classes):
    existing_mask = true_classes == 1
    #descending in probability
    ordered_indices = np.flip(probs.argsort())
    ordered_existing_mask = existing_mask[ordered_indices]
    existing_ranks = np.arange(1,
                                true_classes.shape[0]+1,
                                dtype=float)[ordered_existing_mask]
    MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
    return MRR


def get_MAP(labels,logits):
    logits = softmax(logits.numpy())
    y_scores = logits[:,1]
    y_true = labels.numpy().astype(int)
    return average_precision_score(y_true, y_scores)


def compute_jaccard(edges_snapshot_t, edges_snapshot_t1):
    # Convert arrays to sets of tuples for easier comparison
    set_t = set(map(tuple, edges_snapshot_t))
    set_t1 = set(map(tuple, edges_snapshot_t1))
    
    intersection_size = len(set_t & set_t1)
    union_size = len(set_t | set_t1)
    
    if union_size == 0:
        # If both snapshots have no edges, the similarity is 1.
        return 1.0
    
    jaccard_similarity = intersection_size / union_size
    return jaccard_similarity