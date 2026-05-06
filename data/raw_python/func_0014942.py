def asVerboseContainer(cont, onGet=None, onSet=None, onDel=None):
    """Returns a 'verbose' version of container instance `cont`, that will
       execute `onGet`, `onSet` and `onDel` (if not `None`) every time
       __getitem__, __setitem__ and __delitem__ are called, passing `self`, `key`
       (and `value` in the case of set). E.g:

       >>> l = [1,2,3]
       >>> l = asVerboseContainer(l,
       ...                onGet=lambda s,k:k==2 and prin('Got two:', k),
       ...                onSet=lambda s,k,v:k == v and prin('k == v:', k, v),
       ...                onDel=lambda s,k:k == 1 and prin('Deleting one:', k))
       >>> l
       [1, 2, 3]
       >>> l[1]
       2
       >>> l[2]
       Got two: 2
       3
       >>> l[2] = 22
       >>> l[2] = 2
       k == v: 2 2
       >>> del l[2]
       >>> del l[1]
       Deleting one: 1

    """
    class VerboseContainer(type(cont)):
        if onGet:
            def __getitem__(self, key):
                onGet(self, key)
                return super(VerboseContainer, self).__getitem__(key)
        if onSet:
            def __setitem__(self, key, value):
                onSet(self, key, value)
                return super(VerboseContainer, self).__setitem__(key, value)
        if onDel:
            def __delitem__(self, key):
                onDel(self, key)
                return super(VerboseContainer, self).__delitem__(key)
    return VerboseContainer(cont)