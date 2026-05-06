def one_to_max(array_in):
    """Alter a vector of cluster labels to a dense mapping. 
        Given that this function is herein always called after passing 
        a vector to the function checkcl, one_to_max relies on the assumption 
        that cluster_run does not contain any NaN entries.

    Parameters
    ----------
    array_in : a list or one-dimensional array
        The list of cluster IDs to be processed.
    
    Returns
    -------
    result : one-dimensional array
        A massaged version of the input vector of cluster identities.
    """
    
    x = np.asanyarray(array_in)
    N_in = x.size
    array_in = x.reshape(N_in)    

    sorted_array = np.sort(array_in)
    sorting_indices = np.argsort(array_in)

    last = np.nan
    current_index = -1
    for i in range(N_in):
        if last != sorted_array[i] or np.isnan(last):
            last = sorted_array[i]
            current_index += 1

        sorted_array[i] = current_index

    result = np.empty(N_in, dtype = int)
    result[sorting_indices] = sorted_array

    return result