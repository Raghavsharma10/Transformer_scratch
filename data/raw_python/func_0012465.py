def _compound_column_value(k1, k2):
        """
        Like :py:meth:`~._column_value` but collapses two unknowns into one.

        :param k1: first (top-level) value
        :param k2: second (bottom-level) value
        :return: display key
        :rtype: str
        """
        k1 = ProjectStats._column_value(k1)
        k2 = ProjectStats._column_value(k2)
        if k1 == 'unknown' and k2 == 'unknown':
            return 'unknown'
        return '%s %s' % (k1, k2)