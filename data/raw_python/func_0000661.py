def checks(similarities, verbose = False):
    """Check that a matrix is a proper similarity matrix and bring 
        appropriate changes if applicable.

    Parameters
    ----------
    similarities : array of shape (n_samples, n_samples)
        A matrix of pairwise similarities between (sub)-samples of the data-set. 

    verbose : Boolean, optional (default = False)
        Alerts of any issue with the similarities matrix provided
        and of any step possibly taken to remediate such problem.
    """
    
    if similarities.size == 0:
        raise ValueError("\nERROR: Cluster_Ensembles: checks: the similarities "
                         "matrix provided as input happens to be empty.\n")
    elif np.where(np.isnan(similarities))[0].size != 0:
        raise ValueError("\nERROR: Cluster_Ensembles: checks: input similarities "
                         "matrix contains at least one 'NaN'.\n")
    elif np.where(np.isinf(similarities))[0].size != 0:
        raise ValueError("\nERROR: Cluster_Ensembles: checks: at least one infinite entry "
                         "detected in input similarities matrix.\n")
    else:
        if np.where(np.logical_not(np.isreal(similarities)))[0].size != 0:
            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: complex entries found "
                      "in the similarities matrix.")

            similarities = similarities.real

            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: "
                      "truncated to their real components.")

        if similarities.shape[0] != similarities.shape[1]:
            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: non-square matrix provided.")

            N_square = min(similarities.shape)
            similarities = similarities[:N_square, :N_square]

            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: using largest square sub-matrix.")

        max_sim = np.amax(similarities)
        min_sim = np.amin(similarities)
        if max_sim > 1 or min_sim < 0:
            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: strictly negative "
                      "or bigger than unity entries spotted in input similarities matrix.")

            indices_too_big = np.where(similarities > 1) 
            indices_negative = np.where(similarities < 0)
            similarities[indices_too_big] = 1.0
            similarities[indices_negative] = 0.0

            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: done setting them to "
                      "the lower or upper accepted values.")     

        if not np.allclose(similarities, np.transpose(similarities)):
            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: non-symmetric input "
                      "similarities matrix.")

            similarities = np.divide(similarities + np.transpose(similarities), 2.0)

            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: now symmetrized.")

        if not np.allclose(np.diag(similarities), np.ones(similarities.shape[0])):
            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: the self-similarities "
                      "provided as input are not all of unit value.")

            similarities[np.diag_indices(similarities.shape[0])] = 1

            if verbose:
                print("\nINFO: Cluster_Ensembles: checks: issue corrected.")