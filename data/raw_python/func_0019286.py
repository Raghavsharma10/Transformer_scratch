def seconds_left(self):
        """Remaining part of the year in seconds.

        In the first example, only one minute and thirty seconds of the year
        remain:

        >>> from hydpy.core.timetools import TOY
        >>> TOY('12_31_23_58_30').seconds_left
        90

        The second example shows that the 29th February is generally included:

        >>> TOY('2').seconds_left
        28944000
        """
        return int((self._ENDDATE.datetime -
                    Date(self).datetime).total_seconds())