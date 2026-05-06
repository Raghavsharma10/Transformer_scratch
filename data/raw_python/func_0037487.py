def _is_cif(string):
        """Test if input string is in CIF format.
        
        :param string: Input string.
        :type string: :py:class:`str` or :py:class:`bytes`
        :return: Input string if in CIF format or False otherwise.
        :rtype: :py:class:`str` or :py:obj:`False` 
        """
        if (string[0:5] == u"data_" and u"_entry.id" in string) or (string[0:5] == b"data_" and b"_entry.id" in string):
            return string
        return False