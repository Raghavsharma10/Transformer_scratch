def orderBy(self, field, direct="desc"):
        """
        Allows for the results to be ordered by a specific field. If given,
        direction can be set with passing an additional argument in the form
        of "asc" or "desc"
        """
        if direct == "desc":
            self._query = self._query.order_by(r.desc(field))
        else:
            self._query = self._query.order_by(r.asc(field))

        return self