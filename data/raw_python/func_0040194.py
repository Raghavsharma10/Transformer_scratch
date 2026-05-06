def pageassert(func):
    '''
    Decorator that assert page number
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args[0] < 1 or args[0] > 40:
            raise ValueError('Page Number not found')
        return func(*args, **kwargs)
    return wrapper