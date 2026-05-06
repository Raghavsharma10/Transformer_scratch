def json_repr(self, minimal=False):
        """Construct a JSON-friendly representation of the object.

        :param bool minimal: [ignored]

        :rtype: list
        """
        if self.value:
            return [self.field, self.operator, self.value]
        else:
            return [self.field, self.operator]