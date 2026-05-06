def memoize(fn):
    '''Cache the results of a function that only takes positional arguments.'''

    cache = {}

    @wraps(fn)
    def wrapped_function(*args):
        if args in cache:
            return cache[args]

        else:
            result = fn(*args)
            cache[args] = result
            return result

    return wrapped_function