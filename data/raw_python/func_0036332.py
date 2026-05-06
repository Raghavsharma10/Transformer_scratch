def matplotlibensure(func):
    """If matplotlib isn't installed, this decorator alerts the user and 
    suggests how one might obtain the package."""  
    @wraps(func)
    def wrap(*args):
        if MPLINSTALLED == False:
            raise ImportError(msg)
        
        return func(*args)   
        
    return wrap