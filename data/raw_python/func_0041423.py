def strip_rows(array,invalid=None):
    '''takes a ``list`` of ``list``s and removes corresponding indices containing the
    invalid value (default ``None``). '''
    array = np.array(array)
    none_indices = np.where(np.any(np.equal(array,invalid),axis=0))
    return tuple(np.delete(array,none_indices,axis=1))