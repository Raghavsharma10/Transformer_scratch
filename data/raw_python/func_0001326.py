def from_env(parser_modules: t.Optional[t.Union[t.List[str], t.Tuple[str]]] = DEFAULT_PARSER_MODULES,
                 env: t.Optional[t.Dict[str, str]] = None,
                 silent: bool = False,
                 suppress_logs: bool = False,
                 extra: t.Optional[dict] = None) -> 'ConfigLoader':
        """
        Creates an instance of :class:`~django_docker_helpers.config.ConfigLoader`
        with parsers initialized from environment variables.

        By default it tries to initialize all bundled parsers.
        Parsers may be customized with ``parser_modules`` argument or ``CONFIG__PARSERS`` environment variable.
        Environment variable has a priority over the method argument.

        :param parser_modules: a list of dot-separated module paths
        :param env: a dict with environment variables, default is ``os.environ``
        :param silent: passed to :class:`~django_docker_helpers.config.ConfigLoader`
        :param suppress_logs: passed to :class:`~django_docker_helpers.config.ConfigLoader`
        :param extra: pass extra arguments to *every* parser
        :return: an instance of :class:`~django_docker_helpers.config.ConfigLoader`

        Example:
        ::

            env = {
                'CONFIG__PARSERS': 'EnvironmentParser,RedisParser,YamlParser',
                'ENVIRONMENTPARSER__SCOPE': 'nested',
                'YAMLPARSER__CONFIG': './tests/data/config.yml',
                'REDISPARSER__HOST': 'wtf.test',
                'NESTED__VARIABLE': 'i_am_here',
            }

            loader = ConfigLoader.from_env(env=env)
            assert [type(p) for p in loader.parsers] == [EnvironmentParser, RedisParser, YamlParser]
            assert loader.get('variable') == 'i_am_here', 'Ensure env copied from ConfigLoader'

            loader = ConfigLoader.from_env(parser_modules=['EnvironmentParser'], env={})

        """
        env = env or os.environ
        extra = extra or {}
        environment_parser = EnvironmentParser(scope='config', env=env)
        silent = environment_parser.get('silent', silent, coerce_type=bool)
        suppress_logs = environment_parser.get('suppress_logs', suppress_logs, coerce_type=bool)

        env_parsers = environment_parser.get('parsers', None, coercer=comma_str_to_list)
        if not env_parsers and not parser_modules:
            raise ValueError('Must specify `CONFIG__PARSERS` env var or `parser_modules`')

        if env_parsers:
            parser_classes = ConfigLoader.import_parsers(env_parsers)
        else:
            parser_classes = ConfigLoader.import_parsers(parser_modules)

        parsers = []

        for parser_class in parser_classes:
            parser_options = ConfigLoader.load_parser_options_from_env(parser_class, env=env)

            _init_args = inspect.getfullargspec(parser_class.__init__).args
            # add extra args if parser's __init__ can take it it
            if 'env' in _init_args:
                parser_options['env'] = env

            for k, v in extra.items():
                if k in _init_args:
                    parser_options[k] = v

            parser_instance = parser_class(**parser_options)
            parsers.append(parser_instance)

        return ConfigLoader(parsers=parsers, silent=silent, suppress_logs=suppress_logs)