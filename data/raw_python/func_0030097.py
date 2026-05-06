def dictmapper(typename, mapping):
    """
    A factory to create `namedtuple`-like classes from a field-to-dict-path
    mapping::

        Person = dictmapper({'person':('person','name')})
        example_dict = {'person':{'name':'John'}}
        john = Person(example_dict)
        assert john.name == 'John'

    If a function is specified as a mapping value instead of a dict "path", it
    will be run with the backing dict as its first argument.
    """
    def init(self, d, *args, **kwargs):
        """
        Initialize `dictmapper` classes with a dict to back getters.
        """
        self._original_dict = d

    def getter_from_dict_path(path):
        if not callable(path) and len(path) < 1:
            raise ValueError('Dict paths should be iterables with at least one'
                             ' key or callable objects that take one argument.')
        def getter(self):
            cur_dict = self._original_dict
            if callable(path):
                return path(cur_dict)
            return dict_value_from_path(cur_dict, path)
        return getter

    prop_mapping = dict((k, property(getter_from_dict_path(v))) 
                        for k, v in mapping.iteritems())
    prop_mapping['__init__'] = init
    return type(typename, tuple(), prop_mapping)