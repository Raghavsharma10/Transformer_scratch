def from_json(cls, obj):
        """Construct a MarathonConstraint from a parsed response.

        :param dict attributes: object attributes from parsed response

        :rtype: :class:`MarathonConstraint`
        """
        if len(obj) == 2:
            (field, operator) = obj
            return cls(field, operator)
        if len(obj) > 2:
            (field, operator, value) = obj
            return cls(field, operator, value)