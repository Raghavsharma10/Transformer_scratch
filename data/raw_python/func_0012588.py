def dict_fun(data, function):
    """
    Apply a function to all values in a dictionary, return a dictionary with
    results.

    Parameters
    ----------
    data : dict
        a dictionary whose values are adequate input to the second argument
        of this function. 
    function : function
        a function that takes one argument

    Returns
    -------
    a dictionary with the same keys as data, such that
    result[key] = function(data[key])
    """
    return dict((k, function(v)) for k, v in list(data.items()))