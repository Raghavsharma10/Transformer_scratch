def _jsTypeof(o):
    '''Return a string similar to JS's typeof.'''
    if o == None:
        return 'object'
    elif o == Undefined:
        return 'undefined'
    elif isinstance(o, bool):
        return 'boolean'
    if isinstance(o, int) or isinstance(o, float):
        return 'number'
    elif isinstance(o, list) or isinstance(o, dict):
        return 'object'
    elif isinstance(o, stringtype):
        return 'string'
    raise ValueError('Unknown type for object %s (%s)' % (o, type(o)))