def load_parser_options_from_env(
            parser_class: t.Type[BaseParser],
            env: t.Optional[t.Dict[str, str]] = None) -> t.Dict[str, t.Any]:
        """
        Extracts arguments from ``parser_class.__init__`` and populates them from environment variables.

        Uses ``__init__`` argument type annotations for correct type casting.

        .. note::

            Environment variables should be prefixed with ``<UPPERCASEPARSERCLASSNAME>__``.

        :param parser_class: a subclass of :class:`~django_docker_helpers.config.backends.base.BaseParser`
        :param env: a dict with environment variables, default is ``os.environ``
        :return: parser's ``__init__`` arguments dict mapping

        Example:
        ::

            env = {
                'REDISPARSER__ENDPOINT': 'go.deep',
                'REDISPARSER__HOST': 'my-host',
                'REDISPARSER__PORT': '66',
            }

            res = ConfigLoader.load_parser_options_from_env(RedisParser, env)
            assert res == {'endpoint': 'go.deep', 'host': 'my-host', 'port': 66}
        """
        env = env or os.environ
        sentinel = object()
        spec: inspect.FullArgSpec = inspect.getfullargspec(parser_class.__init__)
        environment_parser = EnvironmentParser(scope=parser_class.__name__.upper(), env=env)

        stop_args = ['self']
        safe_types = [int, bool, str]

        init_args = {}

        for arg_name in spec.args:
            if arg_name in stop_args:
                continue

            type_hint = spec.annotations.get(arg_name)
            coerce_type = None

            if type_hint in safe_types:
                coerce_type = type_hint
            elif hasattr(type_hint, '__args__'):
                if len(type_hint.__args__) == 1:  # one type
                    if type_hint.__args__[0] in safe_types:
                        coerce_type = type_hint.__args__[0]
                elif len(type_hint.__args__) == 2:  # t.Optional
                    try:
                        _args = list(type_hint.__args__)
                        _args.remove(type(None))
                        if _args[0] in safe_types:
                            coerce_type = _args[0]
                    except ValueError:
                        pass

            val = environment_parser.get(arg_name, sentinel, coerce_type=coerce_type)
            if val is sentinel:
                continue

            init_args[arg_name] = val

        return init_args