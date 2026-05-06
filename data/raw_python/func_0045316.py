def __clip(val, minimum, maximum):
        """
        
        :param val: input value 
        :param minimum: min value
        :param maximum: max value
        :return: val clipped to range [minimum, maximum]
        """
        if val is None or minimum is None or maximum is None:
            return None
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val