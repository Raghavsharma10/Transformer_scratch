def add_new_init_method(cls):
    """
    Replace the existing cls.__init__() method with a new one which
    also initialises the _dependent_generators attribute to an empty list.
    """

    orig_init = cls.__init__

    def new_init(self, *args, **kwargs):
        self._dependent_generators = []
        orig_init(self, *args, **kwargs)

    cls.__init__ = new_init