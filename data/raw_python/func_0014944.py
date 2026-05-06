def d2attrs(*args, **kwargs):
    """Utility function to remove ``**kwargs`` parsing boiler-plate in
       ``__init__``:

        >>> kwargs = dict(name='Bill', age=51, income=1e7)
        >>> self = ezstruct(); d2attrs(kwargs, self, 'income', 'name'); self
        ezstruct(income=10000000.0, name='Bill')
        >>> self = ezstruct(); d2attrs(kwargs, self, 'income', age=0, bloodType='A'); self
        ezstruct(age=51, bloodType='A', income=10000000.0)

        To set all keys from ``kwargs`` use:

        >>> self = ezstruct(); d2attrs(kwargs, self, 'all!'); self
        ezstruct(age=51, income=10000000.0, name='Bill')
    """
    (d, self), args = args[:2], args[2:]
    if args[0] == 'all!':
        assert len(args) == 1
        for k in d: setattr(self, k, d[k])
    else:
        if len(args) != len(set(args)) or set(kwargs) & set(args):
            raise ValueError('Duplicate keys: %s' %
                             list(notUnique(args)) + list(set(kwargs) & set(args)))
        for k in args:
            if k in kwargs: raise ValueError('%s specified twice' % k)
            setattr(self, k, d[k])
        for dk in kwargs:
            setattr(self, dk, d.get(dk, kwargs[dk]))