def load_formatter_fn(formatter):
    '''
    >>> load_formatter_fn('logagg.formatters.basescript') #doctest: +ELLIPSIS
    <function basescript at 0x...>
    '''
    obj = util.load_object(formatter)
    if not hasattr(obj, 'ispartial'):
        obj.ispartial = util.ispartial
    return obj