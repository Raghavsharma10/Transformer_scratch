def get(self,
            variable_path: str,
            default: t.Optional[t.Any] = None,
            coerce_type: t.Optional[t.Type] = None,
            coercer: t.Optional[t.Callable] = None,
            **kwargs):
        """
        Reads a value of ``variable_path`` from consul kv storage.

        :param variable_path: a delimiter-separated path to a nested value
        :param default: default value if there's no object by specified path
        :param coerce_type: cast a type of a value to a specified one
        :param coercer: perform a type casting with specified callback
        :param kwargs: additional arguments inherited parser may need
        :return: value or default

        :raises config.exceptions.KVStorageKeyDoestNotExist: if specified ``endpoint`` does not exists

        :raises config.exceptions.KVStorageValueIsEmpty: if specified ``endpoint`` does not contain a config
        """

        return self.inner_parser.get(
            variable_path,
            default=default,
            coerce_type=coerce_type,
            coercer=coercer,
            **kwargs,
        )