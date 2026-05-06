def _cut_tree(tree, n_clusters, membs):
    """ Cut the tree to get desired number of clusters as n_clusters
            2 <= n_desired <= n_clusters
    """
    ## starting from root,
    ## a node is added to the cut_set or 
    ## its children are added to node_set
    assert(n_clusters >= 2)
    assert(n_clusters <= len(tree.leaves()))

    cut_centers = dict() #np.empty(shape=(n_clusters, ndim), dtype=float)
    
    for i in range(n_clusters-1):
        if i==0:
            search_set = set(tree.children(0))
            node_set,cut_set = set(), set()
        else:
            search_set = node_set.union(cut_set)
            node_set,cut_set = set(), set()

        if i+2 == n_clusters:
            cut_set = search_set
        else:
            for _ in range(len(search_set)):
                n = search_set.pop()
            
                if n.data['ilev'] is None or n.data['ilev']>i+2:
                    cut_set.add(n)
                else:
                    nid = n.identifier
                    if n.data['ilev']-2==i:
                        node_set = node_set.union(set(tree.children(nid)))
   
    conv_membs = membs.copy()
    for node in cut_set:
        nid = node.identifier
        label = node.data['label']
        cut_centers[label] = node.data['center']
        sub_leaves = tree.leaves(nid)
        for leaf in sub_leaves:
            indx = np.where(conv_membs == leaf)[0]
            conv_membs[indx] = nid

    return(conv_membs, cut_centers)