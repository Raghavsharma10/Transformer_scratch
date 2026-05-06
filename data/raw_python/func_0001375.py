def inner_parser(self) -> BaseParser:
        """
        Prepares inner config parser for config stored at ``endpoint``.

        :return: an instance of :class:`~django_docker_helpers.config.backends.base.BaseParser`

        :raises config.exceptions.KVStorageValueIsEmpty: if specified ``endpoint`` does not contain a config
        """
        if self._inner_parser is not None:
            return self._inner_parser

        config = self.client.get(self.endpoint)
        if not config:
            raise KVStorageValueIsEmpty('Key `{0}` does not exist or value is empty'.format(self.endpoint))

        config = config.decode()

        self._inner_parser = self.inner_parser_class(
            config=io.StringIO(config),
            path_separator=self.path_separator,
            scope=None
        )
        return self._inner_parser