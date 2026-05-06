def enum(**enums):
    """
    A basic enum implementation.

    Usage:
        >>> MY_ENUM = enum(FOO=1, BAR=2)
        >>> MY_ENUM.FOO
        1
        >>> MY_ENUM.BAR
        2
    """
    # Enum values must be hashable to support reverse lookup.
    if not all(isinstance(val, collections.Hashable) for val in _values(enums)):
        raise EnumConstructionException('All enum values must be hashable.')

    # Cheating by maintaining a copy of original dict for iteration b/c iterators are hard.
    # It must be a deepcopy because new.classobj() modifies the original.
    en = copy.deepcopy(enums)
    e = type('Enum', (_EnumMethods,), dict((k, v) for k, v in _items(en)))

    try:
        e.choices = [(v, k) for k, v in sorted(_items(enums), key=itemgetter(1))]  # DEPRECATED
    except TypeError:
        pass
    e.get_id_by_label = e.__dict__.get
    e.get_label_by_id = dict((v, k) for (k, v) in _items(enums)).get

    return e