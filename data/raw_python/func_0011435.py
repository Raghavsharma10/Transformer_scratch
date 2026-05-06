def get_strategy(name_or_cls):
    """Return the strategy identified by its name. If ``name_or_class`` is a class,
    it will be simply returned.
    """
    if isinstance(name_or_cls, six.string_types):
        if name_or_cls not in STRATS:
            raise MutationError("strat is not defined")
        return STRATS[name_or_cls]()

    return name_or_cls()