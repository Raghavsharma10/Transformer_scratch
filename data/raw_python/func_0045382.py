def do_reduce(func_name, *sequence):
    """
    Apply a function of two arguments cumulatively to the items of a sequence,
    from left to right, so as to reduce the sequence to a single value.
    
    Functions may be registered with ``native_tags`` 
    or can be ``builtins`` or from the ``operator`` module
    
    Syntax::
    
        {% reduce [function] [sequence] %}        
        {% reduce [function] [item1 item2 ...] %}
    
    For example::
    
        {% reduce add 1 2 3 4 5 %}
        
    calculates::
    
        ((((1+2)+3)+4)+5) = 15
    """
    if len(sequence)==1:
        sequence = sequence[0]
    return reduce(get_func(func_name), sequence)