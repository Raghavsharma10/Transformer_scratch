def get(cls, name=None):
        """Gets the application wide depot instance.

        Might return ``None`` if :meth:`configure` has not been
        called yet.

        """
        if name is None:
            name = cls._default_depot

        name = cls.resolve_alias(name)  # resolve alias
        return cls._depots.get(name)