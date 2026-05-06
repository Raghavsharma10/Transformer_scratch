def _todict(cls):
        """ generate a dict keyed by value """
        return dict((getattr(cls, attr), attr) for attr in dir(cls) if not attr.startswith('_'))