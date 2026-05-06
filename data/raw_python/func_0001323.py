def get(self,
            variable_path: str,
            default: t.Optional[t.Any] = None,
            coerce_type: t.Optional[t.Type] = None,
            coercer: t.Optional[t.Callable] = None,
            required: bool = False,
            **kwargs):
        """
        Tries to read a ``variable_path`` from each of the passed parsers.
        It stops if read was successful and returns a retrieved value.
        If none of the parsers contain a value for the specified path it returns ``default``.

        :param variable_path: a path to variable in config
        :param default: a default value if ``variable_path`` is not present anywhere
        :param coerce_type: cast a result to a specified type
        :param coercer: perform the type casting with specified callback
        :param required: raise ``RequiredValueIsEmpty`` if no ``default`` and no result
        :param kwargs: additional options to all parsers
        :return: **the first successfully read** value from the list of parser instances or ``default``

        :raises config.exceptions.RequiredValueIsEmpty: if nothing is read,``required``
         flag is set, and there's no ``default`` specified
        """

        for p in self.parsers:
            try:
                val = p.get(
                    variable_path, default=self.sentinel,
                    coerce_type=coerce_type, coercer=coercer,
                    **kwargs
                )
                if val != self.sentinel:
                    self.enqueue(variable_path, p, val)
                    return val
            except Exception as e:
                if not self.silent:
                    raise
                if self.suppress_logs:
                    continue
                self.logger.error('Parser {0} cannot get key `{1}`: {2}'.format(
                    p.__class__.__name__,
                    variable_path,
                    str(e)
                ))

        self.enqueue(variable_path, value=default)

        if not default and required:
            raise exceptions.RequiredValueIsEmpty(
                'No default provided and no value read for `{0}`'.format(variable_path))

        return default