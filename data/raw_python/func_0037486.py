def _is_nmrstar(string):
        """Test if input string is in NMR-STAR format.

        :param string: Input string.
        :type string: :py:class:`str` or :py:class:`bytes`
        :return: Input string if in NMR-STAR format or False otherwise.
        :rtype: :py:class:`str` or :py:obj:`False`
        """
        if (string[0:5] == u"data_" and u"save_" in string) or (string[0:5] == b"data_" and b"save_" in string):
            return string
        return False