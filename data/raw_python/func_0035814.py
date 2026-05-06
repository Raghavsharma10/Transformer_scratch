def _to_camel_case(string):
        """Return a camel cased version of the input string.

        Args:
            string (str): A snake cased string.

        Returns:
            str: A camel cased string.
        """
        components = string.split('_')
        return '%s%s' % (
            components[0],
            ''.join(c.title() for c in components[1:]),
        )