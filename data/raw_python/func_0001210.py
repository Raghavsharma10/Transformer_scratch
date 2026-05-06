def get(self,
            variable_path: str,
            default: t.Optional[t.Any] = None,
            coerce_type: t.Optional[t.Type] = None,
            coercer: t.Optional[t.Callable] = None,
            **kwargs):
        """
        Reads a value of ``variable_path`` from environment.

        If ``coerce_type`` is ``bool`` and no ``coercer`` specified, ``coerces`` forced to be
        :func:`~django_docker_helpers.utils.coerce_str_to_bool`

        :param variable_path: a delimiter-separated path to a nested value
        :param default: default value if there's no object by specified path
        :param coerce_type: cast a type of a value to a specified one
        :param coercer: perform a type casting with specified callback
        :param kwargs: additional arguments inherited parser may need
        :return: value or default
        """

        var_name = self.get_env_var_name(variable_path)
        val = self.env.get(var_name, self.sentinel)
        if val is self.sentinel:
            return default

        # coerce to bool with default env coercer if no coercer specified
        if coerce_type and coerce_type is bool and not coercer:
            coercer = coerce_str_to_bool

        return self.coerce(val, coerce_type=coerce_type, coercer=coercer)