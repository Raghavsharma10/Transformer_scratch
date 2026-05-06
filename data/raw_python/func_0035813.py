def _to_snake_case(string):
        """Return a snake cased version of the input string.

        Args:
            string (str): A camel cased string.

        Returns:
            str: A snake cased string.
        """
        sub_string = r'\1_\2'
        string = REGEX_CAMEL_FIRST.sub(sub_string, string)
        return REGEX_CAMEL_SECOND.sub(sub_string, string).lower()