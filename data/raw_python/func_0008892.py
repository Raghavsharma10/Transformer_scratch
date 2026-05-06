def _get_flat_ids(assigned):
    """
    This is a helper function to recover the coordinates of regions that have
    been labeled within an image. This function efficiently computes the
    coordinate of all regions and returns the information in a memory-efficient
    manner.

    Parameters
    -----------
    assigned : ndarray[ndim=2, dtype=int]
        The labeled image. For example, the result of calling
        scipy.ndimage.label on a binary image

    Returns
    --------
    I : ndarray[ndim=1, dtype=int]
        Array of 1d coordinate indices of all regions in the image
    region_ids : ndarray[shape=[n_features + 1], dtype=int]
        Indexing array used to separate the coordinates of the different
        regions. For example, region k has xy coordinates of
        xy[region_ids[k]:region_ids[k+1], :]
    labels : ndarray[ndim=1, dtype=int]
        The labels of the regions in the image corresponding to the coordinates
        For example, assigned.ravel()[I[k]] == labels[k]
    """
    # MPU optimization:
    # Let's segment the regions and store in a sparse format
    # First, let's use where once to find all the information we want
    ids_labels = np.arange(len(assigned.ravel()), 'int64')
    I = ids_labels[assigned.ravel().astype(bool)]
    labels = assigned.ravel()[I]
    # Now sort these arrays by the label to figure out where to segment
    sort_id = np.argsort(labels)
    labels = labels[sort_id]
    I = I[sort_id]
    # this should be of size n_features-1
    region_ids = np.where(labels[1:] - labels[:-1] > 0)[0] + 1
    # This should be of size n_features + 1
    region_ids = np.concatenate(([0], region_ids, [len(labels)]))

    return [I, region_ids, labels]