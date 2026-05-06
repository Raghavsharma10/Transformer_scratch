def collapsesum(data_frame, by = None, var = None):
    '''
    Pour une variable, fonction qui calcule la moyenne pondérée au sein de chaque groupe.
    '''
    assert by is not None
    assert var is not None
    grouped = data_frame.groupby([by])
    return grouped.apply(lambda x: weighted_sum(groupe = x, var =var))