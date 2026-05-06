def is_literal_eval(node_or_string) -> tuple:
    """
    Check if an expresion can be literal_eval.

    ----------
    node_or_string : 
        Input

    Returns
    -------
    tuple
        (bool,python object)

        
        If it can be literal_eval the python object is returned. Otherwise None it is returned.
        
    >>> is_literal_eval('[1,2,3]')
    (True, [1, 2, 3])
    >>> is_literal_eval('a')
    (False, None)
    """
    try:
        obj=ast.literal_eval(node_or_string)
        return (True, obj)
    except:
        return (False, None)