def _bisect_kmeans(X, n_clusters, n_trials, max_iter, tol):
    """ Apply Bisecting Kmeans clustering
        to reach n_clusters number of clusters
    """
    membs = np.empty(shape=X.shape[0], dtype=int)
    centers = dict() #np.empty(shape=(n_clusters,X.shape[1]), dtype=float)
    sse_arr = dict() #-1.0*np.ones(shape=n_clusters, dtype=float)

    ## data structure to store cluster hierarchies
    tree = treelib.Tree()
    tree = _add_tree_node(tree, 0, ilev=0, X=X) 

    km = _kmeans.KMeans(n_clusters=2, n_trials=n_trials, max_iter=max_iter, tol=tol)
    for i in range(1,n_clusters):
        sel_clust_id,sel_memb_ids = _select_cluster_2_split(membs, tree)
        X_sub = X[sel_memb_ids,:]
        km.fit(X_sub)

        #print("Bisecting Step %d    :"%i, sel_clust_id, km.sse_arr_, km.centers_)
        ## Updating the clusters & properties
        #sse_arr[[sel_clust_id,i]] = km.sse_arr_
        #centers[[sel_clust_id,i]] = km.centers_
        tree = _add_tree_node(tree, 2*i-1, i, \
                              size=np.sum(km.labels_ == 0), center=km.centers_[0], \
                              sse=km.sse_arr_[0], parent= sel_clust_id)
        tree = _add_tree_node(tree, 2*i,   i, \
                             size=np.sum(km.labels_ == 1), center=km.centers_[1], \
                             sse=km.sse_arr_[1], parent= sel_clust_id)

        pred_labels = km.labels_
        pred_labels[np.where(pred_labels == 1)[0]] = 2*i
        pred_labels[np.where(pred_labels == 0)[0]] = 2*i - 1
        #if sel_clust_id == 1:
        #    pred_labels[np.where(pred_labels == 0)[0]] = sel_clust_id
        #    pred_labels[np.where(pred_labels == 1)[0]] = i
        #else:
        #    pred_labels[np.where(pred_labels == 1)[0]] = i
        #    pred_labels[np.where(pred_labels == 0)[0]] = sel_clust_id

        membs[sel_memb_ids] = pred_labels


    for n in tree.leaves():
        label = n.data['label']
        centers[label] = n.data['center']
        sse_arr[label] = n.data['sse']

    return(centers, membs, sse_arr, tree)