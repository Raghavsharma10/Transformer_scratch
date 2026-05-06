def seconds_passed(self):
        """Amount of time passed in seconds since the beginning of the year.

        In the first example, the year is only one minute and thirty seconds
        old:

        >>> from hydpy.core.timetools import TOY
        >>> TOY('1_1_0_1_30').seconds_passed
        90

        The second example shows that the 29th February is generally included:

        >>> TOY('3').seconds_passed
        5184000
        """
        return int((Date(self).datetime -
                    self._STARTDATE.datetime).total_seconds())