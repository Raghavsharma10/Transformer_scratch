def dump(obj, attributes = True, _refset = None):
    "Show full value of a data object"
    if _refset is None:
        _refset = set()
    if obj is None:
        return None
    elif isinstance(obj, DataObject):        
        if id(obj) in _refset:
            attributes = False
        else:
            _refset.add(id(obj))
        cls = type(obj)
        clsname = getattr(cls, '__module__', '<unknown>') + '.' + getattr(cls, '__name__', '<unknown>')
        baseresult = {'_type': clsname, '_key': obj.getkey()}
        if not attributes:
            return baseresult
        else:
            baseresult.update((k,dump(v, attributes, _refset)) for k,v in vars(obj).items() if k[:1] != '_')
            _refset.remove(id(obj))
        return baseresult
    elif isinstance(obj, ReferenceObject):
        if obj._ref is not None:
            return dump(obj._ref, attributes, _refset)
        else:
            return {'_ref':obj.getkey()}
    elif isinstance(obj, WeakReferenceObject):
        return {'_weakref':obj.getkey()}
    elif isinstance(obj, DataObjectSet):
        return dump(list(obj.dataset()))
    elif isinstance(obj, dict):
        return dict((k, dump(v, attributes, _refset)) for k,v in obj.items())
    elif isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
        return [dump(v, attributes, _refset) for v in obj]
    else:
        return obj