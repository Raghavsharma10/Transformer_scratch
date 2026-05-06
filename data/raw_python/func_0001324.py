def import_parsers(parser_modules: t.Iterable[str]) -> t.Generator[t.Type[BaseParser], None, None]:
        """
        Resolves and imports all modules specified in ``parser_modules``. Short names from the local scope
        are supported (the scope is ``django_docker_helpers.config.backends``).

        :param parser_modules: a list of dot-separated module paths
        :return: a generator of [probably] :class:`~django_docker_helpers.config.backends.base.BaseParser`

        Example:
        ::

            parsers = list(ConfigLoader.import_parsers([
                'EnvironmentParser',
                'django_docker_helpers.config.backends.YamlParser'
            ]))
            assert parsers == [EnvironmentParser, YamlParser]
        """
        for import_path in parser_modules:
            path_parts = import_path.rsplit('.', 1)
            if len(path_parts) == 2:
                mod_path, parser_class_name = path_parts
            else:
                mod_path = DEFAULT_PARSER_MODULE_PATH
                parser_class_name = import_path

            yield import_from(mod_path, parser_class_name)