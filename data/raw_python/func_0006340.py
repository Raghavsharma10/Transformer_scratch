def unique_row(array, use_columns=None, selected_columns_only=False):
    '''Takes a numpy array and returns the array reduced to unique rows. If columns are defined only these columns are taken to define a unique row.
    The returned array can have all columns of the original array or only the columns defined in use_columns.
    Parameters
    ----------
    array : numpy.ndarray
    use_columns : list
        Index of columns to be used to define a unique row
    selected_columns_only : bool
        If true only the columns defined in use_columns are returned

    Returns
    -------
    numpy.ndarray
    '''
    if array.dtype.names is None:  # normal array has no named dtype
        if use_columns is not None:
            a_cut = array[:, use_columns]
        else:
            a_cut = array
        if len(use_columns) > 1:
            b = np.ascontiguousarray(a_cut).view(np.dtype((np.void, a_cut.dtype.itemsize * a_cut.shape[1])))
        else:
            b = np.ascontiguousarray(a_cut)
        _, index = np.unique(b, return_index=True)
        if not selected_columns_only:
            return array[np.sort(index)]  # sort to preserve order
        else:
            return a_cut[np.sort(index)]  # sort to preserve order
    else:  # names for dtype founnd --> array is recarray
        names = list(array.dtype.names)
        if use_columns is not None:
            new_names = [names[i] for i in use_columns]
        else:
            new_names = names
        a_cut, index = np.unique(array[new_names], return_index=True)
        if not selected_columns_only:
            return array[np.sort(index)]  # sort to preserve order
        else:
            return array[np.sort(index)][new_names]