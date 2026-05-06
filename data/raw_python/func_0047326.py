def sort(self, field, direction="asc"):
        """
        Adds sort criteria.
        """
        if not isinstance(field, basestring):
            raise ValueError("Field should be a string")
        if direction not in ["asc", "desc"]:
            raise ValueError("Sort direction should be `asc` or `desc`")

        self.sorts.append({field: direction})