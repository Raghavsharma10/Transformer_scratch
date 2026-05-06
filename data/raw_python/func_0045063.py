def apply_markup(value, arg=None):
    """
    Applies text-to-HTML conversion.
    
    Takes an optional argument to specify the name of a filter to use.
    
    """
    if arg is not None:
        return formatter(value, filter_name=arg)
    return formatter(value)