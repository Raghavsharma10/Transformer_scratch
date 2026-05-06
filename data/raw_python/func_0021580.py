def vector_str(p, decimal_places=2, print_zero=True):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([' ' if not print_zero and a == 0 else style.format(a) for a in p]))