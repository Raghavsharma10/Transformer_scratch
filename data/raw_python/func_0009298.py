def _pct_encode_uri(cls, c):
        """
        Return the appropriate percent-encoding of character c (URI string).
        Certain characters are returned without encoding.

        :param string c: Character to check
        :returns: Encoded character as URI
        :rtype: string

        TEST:

        >>> c = '.'
        >>> CPEComponentSimple._pct_encode_uri(c)
        '.'

        TEST:

        >>> c = '@'
        >>> CPEComponentSimple._pct_encode_uri(c)
        '%40'
        """

        CPEComponentSimple.spechar_to_pce['-'] = c  # bound without encoding
        CPEComponentSimple.spechar_to_pce['.'] = c  # bound without encoding

        return CPEComponentSimple.spechar_to_pce[c]