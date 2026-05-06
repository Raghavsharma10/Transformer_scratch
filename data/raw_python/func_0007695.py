def frange(start, stop, step, digits_to_round=3):
        """
        Works like range for doubles

        :param start: starting value
        :param stop: ending value
        :param step: the increment_value
        :param digits_to_round: the digits to which to round \
        (makes floating-point numbers much easier to work with)
        :return: generator
        """
        while start < stop:
            yield round(start, digits_to_round)
            start += step