def __get_type(self, expectation, options):
        """
        Determining the type of Matcher
        Return: string
        """
        if "is_custom_func" in options.keys():
            setattr(self, "mtest", expectation)
            return "CUSTOMFUNC"
        elif "is_substring" in options.keys():
            return "SUBSTRING"
        elif "is_regex" in options.keys():
            return "REGEX"
        elif isinstance(expectation, type):
            return "TYPE"
        else:
            return "VALUE"