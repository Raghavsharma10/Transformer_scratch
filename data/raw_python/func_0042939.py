def typeOf(cls, expected_type): #pylint: disable=no-self-argument,invalid-name,no-self-use
        """
        (*Type does NOT consider inherited class)
        Matcher.mtest(...) will return True if type(...) == expected_type
        Return: Matcher
        Raise: matcher_type_error
        """
        if isinstance(expected_type, type):
            options = {}
            options["target_type"] = expected_type
            return Matcher("__TYPE__", options)
        ErrorHandler.matcher_type_error(expected_type)