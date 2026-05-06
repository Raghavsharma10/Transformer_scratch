def _unique_hierarchical_string(self):
        """
        Returns:
            str: a representation of time such as::

                '2014/2/23/15/26/8/9877978'

        The last part (microsecond) is needed to avoid duplicates in
        rapid-fire transactions e.g. ``> 1`` edition.

        """
        t = datetime.now()
        return '%s/%s/%s/%s/%s/%s/%s' % (t.year, t.month, t.day, t.hour,
                                         t.minute, t.second, t.microsecond)