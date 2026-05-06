def _has_init_config(self, cls):
        """Return whether the class has a ``config`` parameter in the ``__init__``
        method.

        """
        args = inspect.signature(cls.__init__)
        return self.config_param_name in args.parameters