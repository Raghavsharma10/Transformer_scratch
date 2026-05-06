def _split_scheme(expression):
        """
        Splits the scheme and actual expression

        :param str expression: The expression.

        :rtype: str
        """
        match = re.search(r'^([a-z]+):(.*)$', expression)
        if not match:
            scheme = 'plain'
            actual = expression
        else:
            scheme = match.group(1)
            actual = match.group(2)

        return scheme, actual