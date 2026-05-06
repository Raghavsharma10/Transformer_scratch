def extra(self, **params):
        """
        Set extra query parameters (eg. filter expressions/attributes that don't validate).
        Appends to any previous extras set.

        :rtype: Query
        """
        q = self._clone()
        for key, value in params.items():
            q._extra[key].append(value)
        return q