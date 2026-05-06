def from_string(cls, constraint):
        """
        :param str constraint: The string representation of a constraint

        :rtype: :class:`MarathonConstraint`
        """
        obj = constraint.split(':')
        marathon_constraint = cls.from_json(obj)

        if marathon_constraint:
            return marathon_constraint

        raise ValueError("Invalid string format. "
                         "Expected `field:operator:value`")