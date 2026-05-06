def element_should_not_exist(self, json_string, expr):
        """
        Check that one or more elements, matching [ http://jsonselect.org/ | JSONSelect] expression, don't exist.

        *DEPRECATED* JSON Select query language is outdated and not supported any more.
        Use other keywords of this library to query JSON.

        *Args:*\n
        _json_string_ - JSON string;\n
        _expr_ - JSONSelect expression;\n

        *Raises:*\n
        JsonValidatorError
        """
        value = self.select_elements(json_string, expr)
        if value is not None:
            raise JsonValidatorError('Elements %s exist but should not' % expr)