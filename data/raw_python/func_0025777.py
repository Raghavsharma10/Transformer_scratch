def get_default_value(self, check):
        """
        Given a check, return the default value for the check
        (converted to the right type).

        If the check doesn't specify a default value then a
        ``KeyError`` will be raised.
        """
        fun_name, fun_args, fun_kwargs, default = self._parse_with_caching(check)
        if default is None:
            raise KeyError('Check "%s" has no default value.' % check)
        value = self._handle_none(default)
        if value is None:
            return value
        return self._check_value(value, fun_name, fun_args, fun_kwargs)