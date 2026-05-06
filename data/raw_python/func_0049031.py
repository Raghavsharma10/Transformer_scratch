def to_query(self):
        """
        Returns a json-serializable representation.
        """
        query = {}

        for field_instance in self.fields:
            query.update(field_instance.to_query())

        return query