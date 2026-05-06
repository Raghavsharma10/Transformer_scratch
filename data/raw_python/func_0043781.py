def trim_start(self, string, start, end=None):
        """ Removes the starting substring.
        :param string: The entire string.
        :param start:  The starting substring to be removed.
        :param end:    An optional point in the string, defaults to the strings ending
        :return:       A substring consists of the subsequent substring after the
                         start substring is removed, up to the specified end.
        """
        extract = string[string.find(start) + len(start):end]
        return extract.strip()