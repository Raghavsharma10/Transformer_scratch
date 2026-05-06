def _has_init_name(self, cls):
        """Return whether the class has a ``name`` parameter in the ``__init__``
        method.

        """
        args = inspect.signature(cls.__init__)
        return self.name_param_name in args.parameters