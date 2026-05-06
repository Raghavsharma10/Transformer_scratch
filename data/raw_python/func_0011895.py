def _inner_func_anot(func):
    """must be applied to all inner functions that return contexts.
    
    Wraps all instances of pygame.Surface in the input in Surface"""
    @wraps(func)
    def new_func(*args):
        return func(*_lmap(_wrap_surface, args))
    return new_func