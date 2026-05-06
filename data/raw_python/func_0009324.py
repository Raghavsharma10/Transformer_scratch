def _is_even_wildcards(cls, str, idx):
        """
        Returns True if an even number of escape (backslash) characters
        precede the character at index idx in string str.

        :param string str: string to check
        :returns: True if an even number of escape characters precede
            the character at index idx in string str, False otherwise.
        :rtype: boolean
        """

        result = 0
        while ((idx > 0) and (str[idx - 1] == "\\")):
            idx -= 1
            result += 1

        isEvenNumber = (result % 2) == 0
        return isEvenNumber