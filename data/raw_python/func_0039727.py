def calc(dsets,expr,prefix=None,datum=None):
    ''' returns a string of an inline ``3dcalc``-style expression

    ``dsets`` can be a single string, or list of strings. Each string in ``dsets`` will
    be labeled 'a','b','c', sequentially. The expression ``expr`` is used directly

    If ``prefix`` is not given, will return a 3dcalc string that can be passed to another
    AFNI program as a dataset. Otherwise, will create the dataset with the name ``prefix``'''
    return available_method('calc')(dsets,expr,prefix,datum)