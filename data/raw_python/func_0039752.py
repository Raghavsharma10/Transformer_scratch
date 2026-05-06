def props(cls):
    """
    Class method that returns all defined arguments within the class.
    
    Returns:
      A dictionary containing all action defined arguments (if any).
    """
    return {k:v for (k, v) in inspect.getmembers(cls) if type(v) is Argument}