import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj, test_percent=10., val_percent=20.):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert adj.diagonal().sum() == 0
 
    edges_positive, _, _ = sparse_to_tuple(adj)
    edges_positive = edges_positive[edges_positive[:,1] > edges_positive[:,0],:] # filtering out edges from lower triangle of adjacency matrix
    val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / test_percent))
    num_val = int(np.floor(edges_positive.shape[0] / val_percent))

    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx] # positive test edges
    val_edges = edges_positive[val_edge_idx] # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis=0) # positive train edges

    # the above strategy for sampling without replacement will be slow at sampling negative edges on large graphs, because the pool of negative edges is much much larger due to sparsity
    # therefore we'll use the following strategy:
    # 1. sample N random linear indices from adjacency matrix WITH REPLACEMENT (without replacement is super slow)
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough

    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices

    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')
    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.size, 2*(num_test-len(test_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx,positive_idx,assume_unique=True)]
        idx = idx[~np.in1d(idx,idx_test_edges_false,assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords,axis=0) # note: coords are now sorted lexicographically
        # step 6:
        coords = coords[coords[:,0]!=coords[:,1]]
        # step 7:
        coords = coords[:min(num_test,len(idx))]
        test_edges_false = np.append(test_edges_false,coords,axis=0)
        idx = idx[:min(num_test,len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)


    val_edges_false = np.empty((0,2),dtype='int64')
    idx_val_edges_false = np.empty((0,),dtype='int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.size, 2*(num_val-len(val_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx,positive_idx,assume_unique=True)]
        idx = idx[~np.in1d(idx,idx_test_edges_false,assume_unique=True)]
        idx = idx[~np.in1d(idx,idx_val_edges_false,assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords,axis=0) # note: coords are now sorted lexicographically
        # step 6:
        coords = coords[coords[:,0]!=coords[:,1]]
        # step 7:
        coords = coords[:min(num_val,len(idx))]
        val_edges_false = np.append(val_edges_false,coords,axis=0)
        idx = idx[:min(num_val,len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # sanity checks:
    train_edges_linear = train_edges[:,0]*adj.shape[0] + train_edges[:,1]
    test_edges_linear = test_edges[:,0]*adj.shape[0] + test_edges[:,1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0]+val_edges[:,1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0]+val_edges[:,1], test_edges_linear))

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false