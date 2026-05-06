def timeframe(self, start, end):
        r"""
            When you want to search bugs for a certain time frame.

            :param start:
            :param end:
            :returns: :class:`Search`
        """
        if start:
            self._time_frame['chfieldfrom'] = start
        if end:
            self._time_frame['chfieldto'] = end
        return self