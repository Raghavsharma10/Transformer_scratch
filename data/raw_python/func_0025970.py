def _getAbsoluteTime(self, start, delay):
        """
        Adds the delay in seconds to the start time.
        :param start:
        :param delay:
        :return: a datetimem for the specified point in time.
        """
        return start + datetime.timedelta(days=0, seconds=delay)