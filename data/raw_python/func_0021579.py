def matrix_str(p, decimal_places=2, print_zero=True, label_columns=False):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([(str(i) if label_columns else '') + vector_str(a, decimal_places, print_zero) for i, a in enumerate(p)]))