def get(self, line_number):
        """Return the needle positions or None.

        :param int line_number: the number of the line
        :rtype: list
        :return: the needle positions for a specific line specified by
          :paramref:`line_number` or :obj:`None` if no were given
        """
        if line_number not in self._get_cache:
            self._get_cache[line_number] = self._get(line_number)
        return self._get_cache[line_number]