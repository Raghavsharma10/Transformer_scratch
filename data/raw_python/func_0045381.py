def do_map(func_name, *sequence):
    """
    Return a list of the results of applying the function to the items of
    the argument sequence(s).  
    
    Functions may be registered with ``native_tags`` 
    or can be ``builtins`` or from the ``operator`` module
    
    If more than one sequence is given, the
    function is called with an argument list consisting of the corresponding
    item of each sequence, substituting None for missing values when not all
    sequences have the same length.  If the function is None, return a list of
    the items of the sequence (or a list of tuples if more than one sequence).

    Syntax::
    
        {% map [function] [sequence] %}        
        {% map [function] [item1 item2 ...] %}

    For example::
    
        {% map sha1 hello world %}
        
    calculates::
        
        [sha1(hello), sha1(world)]

    """

    if len(sequence)==1:
        sequence = sequence[0]
    return map(get_func(func_name, False), sequence)