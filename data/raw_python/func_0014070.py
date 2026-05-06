def is_valid_transition(cls, from_value, to_value):
        """ Will check if to_value is a valid transition from from_value. Returns true if it is a valid transition.
        :param from_value: Start transition point
        :param to_value: End transition point
        :type from_value: int
        :type to_value: int
        :return: Success flag
        :rtype: bool
        """
        try:
            return from_value == to_value or from_value in cls.transition_origins(to_value)
        except KeyError:
            return False