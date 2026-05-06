def _is_alphanum(cls, c):
        """
        Returns True if c is an uppercase letter, a lowercase letter,
        a digit or an underscore, otherwise False.

        :param string c: Character to check
        :returns: True if char is alphanumeric or an underscore,
            False otherwise
        :rtype: boolean

        TEST: a wrong character
        >>> c = "#"
        >>> CPEComponentSimple._is_alphanum(c)
        False
        """

        alphanum_rxc = re.compile(CPEComponentSimple._ALPHANUM_PATTERN)
        return (alphanum_rxc.match(c) is not None)