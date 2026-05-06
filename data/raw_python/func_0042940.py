def instanceOf(cls, expected_instance): #pylint: disable=no-self-argument,invalid-name,no-self-use
        """
        (*Instance consider inherited class)
        Matcher.mtest(...) will return True if instance(...) == expected_instance
        Return: Matcher
        Raise: matcher_instance_error
        """
        if not inspect.isclass(expected_instance):
            options = {}
            options["target_type"] = expected_instance
            return Matcher("__INSTANCE__", options)
        ErrorHandler.matcher_instance_error(expected_instance)