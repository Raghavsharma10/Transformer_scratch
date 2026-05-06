def str_to_time(self):
        """
        Formats a XCCDF dateTime string to a datetime object.

        :returns: datetime object.
        :rtype: datetime.datetime
        """

        return datetime(*list(map(int, re.split(r'-|:|T', self.time))))