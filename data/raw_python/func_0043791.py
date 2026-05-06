def is_nested(values):
    '''Check if values is composed only by iterable elements.'''
    return (all(isinstance(item, Iterable) for item in values)
            if isinstance(values, Iterable) else False)