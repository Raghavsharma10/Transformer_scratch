def match(sel, obj, arr=None, bailout_fn=None):
    '''Match a selector to an object, yielding the matched values.

    Args:
        sel: The JSONSelect selector to apply (a string)
        obj: The object against which to apply the selector
        arr: If sel contains ? characters, then the values in this array will
             be safely interpolated into the selector.
        bailout_fn: A callback which takes two parameters, |obj| and |matches|.
             This will be called on every node in obj. If it returns True, the
             search for matches will be aborted below that node. The |matches|
             parameter indicates whether the node matched the selector. This is
             intended to be used as a performance optimization.
    '''
    if arr:
        sel = interpolate(sel, arr)
    sel = parse(sel)[1]
    return _forEach(sel, obj, bailout_fn=bailout_fn)