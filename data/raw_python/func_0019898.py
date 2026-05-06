def induce_correlations(data, corrmat):
    """
    Induce a set of correlations on a column-wise dataset
    
    Parameters
    ----------
    data : 2d-array
        An m-by-n array where m is the number of samples and n is the
        number of independent variables, each column of the array corresponding
        to each variable
    corrmat : 2d-array
        An n-by-n array that defines the desired correlation coefficients
        (between -1 and 1). Note: the matrix must be symmetric and
        positive-definite in order to induce.
    
    Returns
    -------
    new_data : 2d-array
        An m-by-n array that has the desired correlations.
        
    """
    # Create an rank-matrix
    data_rank = np.vstack([rankdata(datai) for datai in data.T]).T

    # Generate van der Waerden scores
    data_rank_score = data_rank / (data_rank.shape[0] + 1.0)
    data_rank_score = norm(0, 1).ppf(data_rank_score)

    # Calculate the lower triangular matrix of the Cholesky decomposition
    # of the desired correlation matrix
    p = chol(corrmat)

    # Calculate the current correlations
    t = np.corrcoef(data_rank_score, rowvar=0)

    # Calculate the lower triangular matrix of the Cholesky decomposition
    # of the current correlation matrix
    q = chol(t)

    # Calculate the re-correlation matrix
    s = np.dot(p, np.linalg.inv(q))

    # Calculate the re-sampled matrix
    new_data = np.dot(data_rank_score, s.T)

    # Create the new rank matrix
    new_data_rank = np.vstack([rankdata(datai) for datai in new_data.T]).T

    # Sort the original data according to new_data_rank
    for i in range(data.shape[1]):
        vals, order = np.unique(
            np.hstack((data_rank[:, i], new_data_rank[:, i])), return_inverse=True
        )
        old_order = order[: new_data_rank.shape[0]]
        new_order = order[-new_data_rank.shape[0] :]
        tmp = data[np.argsort(old_order), i][new_order]
        data[:, i] = tmp[:]

    return data