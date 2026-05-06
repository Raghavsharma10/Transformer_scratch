def inner_parser(self) -> BaseParser:
        """
        Prepares inner config parser for config stored at ``endpoint``.

        :return: an instance of :class:`~django_docker_helpers.config.backends.base.BaseParser`

        :raises config.exceptions.KVStorageKeyDoestNotExist: if specified ``endpoint`` does not exists

        :raises config.exceptions.KVStorageValueIsEmpty: if specified ``endpoint`` does not contain a config
        """
        if self._inner_parser is not None:
            return self._inner_parser

        __index, response_config = self.client.kv.get(self.endpoint, **self.kv_get_opts)
        if not response_config:
            raise KVStorageKeyDoestNotExist('Key does not exist: `{0}`'.format(self.endpoint))

        config = response_config['Value']
        if not config or config is self.sentinel:
            raise KVStorageValueIsEmpty('Read empty config by key `{0}`'.format(self.endpoint))

        config = config.decode()

        self._inner_parser = self.inner_parser_class(
            config=io.StringIO(config),
            path_separator=self.path_separator,
            scope=None
        )
        return self._inner_parser