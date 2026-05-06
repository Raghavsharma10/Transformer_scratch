def pretty_str(p, decimal_places=2, print_zero=True, label_columns=False):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places, print_zero)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places, print_zero, label_columns)
    raise Exception('Invalid array with shape {0}'.format(p.shape))