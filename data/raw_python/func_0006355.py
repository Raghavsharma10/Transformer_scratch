def select_hits(hits_array, condition=None):
    '''Selects the hits with condition.
    E.g.: condition = 'rel_BCID == 7 & event_number < 1000'

    Parameters
    ----------
    hits_array : numpy.array
    condition : string
        A condition that is applied to the hits in numexpr. Only if the expression evaluates to True the hit is taken.

    Returns
    -------
    numpy.array
        hit array with the selceted hits
    '''
    if condition is None:
        return hits_array

    for variable in set(re.findall(r'[a-zA-Z_]+', condition)):
        exec(variable + ' = hits_array[\'' + variable + '\']')

    return hits_array[ne.evaluate(condition)]