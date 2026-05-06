def get(self,
            variable_path: str,
            default: t.Optional[t.Any] = None,
            coerce_type: t.Optional[t.Type] = None,
            coercer: t.Optional[t.Callable] = None,
            **kwargs):
        """
        :param variable_path: a delimiter-separated path to a nested value
        :param default: default value if there's no object by specified path
        :param coerce_type: cast a type of a value to a specified one
        :param coercer: perform a type casting with specified callback
        :param kwargs: additional arguments inherited parser may need
        :return: value or default
        """

        if self.path_separator != self.consul_path_separator:
            variable_path = variable_path.replace(self.path_separator, self.consul_path_separator)

        if self.scope:
            _scope = self.consul_path_separator.join(self.scope.split(self.path_separator))
            variable_path = '{0}/{1}'.format(_scope, variable_path)

        index, data = self.client.kv.get(variable_path, **kwargs)

        if data is None:
            return default

        val = data['Value']
        if val is None:
            # None is present and it is a valid value
            return val

        if val.startswith(self.object_serialize_prefix):
            # since complex data types are yaml-serialized there's no need to coerce anything
            _val = val[len(self.object_serialize_prefix):]
            bundle = self.object_deserialize(_val)
            if bundle == '':  # check for reinforced empty flag
                return self.coerce(bundle, coerce_type=coerce_type, coercer=coercer)
            return bundle

        if isinstance(val, bytes):
            val = val.decode()

        return self.coerce(val, coerce_type=coerce_type, coercer=coercer)