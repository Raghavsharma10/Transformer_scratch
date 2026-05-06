def check(self, check, value, missing=False):
        """
        Usage: check(check, value)

        Arguments:
            check: string representing check to apply (including arguments)
            value: object to be checked
        Returns value, converted to correct type if necessary

        If the check fails, raises a ``ValidateError`` subclass.

        >>> vtor = Validator()
        >>> vtor.check('yoda', '')  # doctest: +SKIP
        Traceback (most recent call last):
        VdtUnknownCheckError: the check "yoda" is unknown.
        >>> vtor.check('yoda()', '')  # doctest: +SKIP
        Traceback (most recent call last):
        VdtUnknownCheckError: the check "yoda" is unknown.

        >>> vtor.check('string(default="")', '', missing=True)
        ''
        """
        fun_name, fun_args, fun_kwargs, default = self._parse_with_caching(check)

        if missing:
            if default is None:
                # no information needed here - to be handled by caller
                raise VdtMissingValue()
            value = self._handle_none(default)

        if value is None:
            return None

        return self._check_value(value, fun_name, fun_args, fun_kwargs)