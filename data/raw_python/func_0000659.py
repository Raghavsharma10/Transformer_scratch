def checkcl(cluster_run, verbose = False):
    """Ensure that a cluster labelling is in a valid format. 

    Parameters
    ----------
    cluster_run : array of shape (n_samples,)
        A vector of cluster IDs for each of the samples selected for a given
        round of clustering. The samples not selected are labelled with NaN.

    verbose : Boolean, optional (default = False)
        Specifies if status messages will be displayed
        on the standard output.

    Returns
    -------
    cluster_run : array of shape (n_samples,)
        The input vector is modified in place, such that invalid values are
        either rejected or altered. In particular, the labelling of cluster IDs
        starts at zero and increases by 1 without any gap left.
    """
    
    cluster_run = np.asanyarray(cluster_run)

    if cluster_run.size == 0:
        raise ValueError("\nERROR: Cluster_Ensembles: checkcl: "
                         "empty vector provided as input.\n")
    elif reduce(operator.mul, cluster_run.shape, 1) != max(cluster_run.shape):
        raise ValueError("\nERROR: Cluster_Ensembles: checkl: "
                         "problem in dimensions of the cluster label vector "
                         "under consideration.\n")
    elif np.where(np.isnan(cluster_run))[0].size != 0:
        raise ValueError("\nERROR: Cluster_Ensembles: checkl: vector of cluster "
                         "labellings provided as input contains at least one 'NaN'.\n")
    else:
        min_label = np.amin(cluster_run)
        if min_label < 0:
            if verbose:
                print("\nINFO: Cluster_Ensembles: checkcl: detected negative values "
                      "as cluster labellings.")

            cluster_run -= min_label

            if verbose:
                print("\nINFO: Cluster_Ensembles: checkcl: "
                      "offset to a minimum value of '0'.")

        x = one_to_max(cluster_run) 
        if np.amax(cluster_run) != np.amax(x):
            if verbose:
                print("\nINFO: Cluster_Ensembles: checkcl: the vector cluster "
                      "labellings provided is not a dense integer mapping.")

            cluster_run = x

            if verbose:
                print("INFO: Cluster_Ensembles: checkcl: brought modification "
                      "to this vector so that its labels range "
                      "from 0 to {0}, included.\n".format(np.amax(cluster_run)))

    return cluster_run