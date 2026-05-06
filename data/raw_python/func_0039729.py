def cluster(dset,min_distance,min_cluster_size,prefix=None):
    '''clusters given ``dset`` connecting voxels ``min_distance``mm away with minimum cluster size of ``min_cluster_size``
    default prefix is ``dset`` suffixed with ``_clust%d``'''
    if prefix==None:
        prefix = nl.suffix(dset,'_clust%d' % min_cluster_size)
    return available_method('cluster')(dset,min_distance,min_cluster_size,prefix)