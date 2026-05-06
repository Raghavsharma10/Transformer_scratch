def find_classes_in_list(klasses, lst):
    """
    Returns a tuple containing an entry corresponding to each of
    the requested class types, where the entry is either the first
    object instance of that type or None of no such instances are
    available.
    
    Example Usage:
    
    find_classes_in_list(
        [Address, Response],
        [<classes.Response...>, <classes.Amount...>, <classes.Address...>])
        
    Produces: (<classes.Address...>, <classes.Response...>)            
    """
    if not isinstance(klasses, list):
        klasses = [klasses]
    return tuple(map(lambda klass: find_class_in_list(klass, lst), klasses))