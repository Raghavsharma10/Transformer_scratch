def construct(self, mapping: dict, **kwargs):
        """
        Construct an object from a mapping

        :param mapping: the constructor definition, with ``__type__`` name and keyword arguments
        :param kwargs: additional keyword arguments to pass to the constructor
        """
        assert '__type__' not in kwargs and '__args__' not in kwargs
        mapping = {**mapping, **kwargs}
        factory_fqdn = mapping.pop('__type__')
        factory = self.load_name(factory_fqdn)
        args = mapping.pop('__args__', [])
        return factory(*args, **mapping)