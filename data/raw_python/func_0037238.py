def parse_grain(grain):
        """ Parse a string to a granularity, e.g. "Day" to InstantTime.day.

        :param grain: a string representing a granularity.
        """
        if not grain:
            return InstantTime.day
        if grain.lower() == 'week':
            return InstantTime.week
        return InstantTime.day