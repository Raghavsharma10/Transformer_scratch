def OR(self):
        """
        Switches default query joiner from " AND " to " OR "

        Returns:
            Self. Queryset object.
        """
        clone = copy.deepcopy(self)
        clone.adapter._QUERY_GLUE = ' OR '
        return clone