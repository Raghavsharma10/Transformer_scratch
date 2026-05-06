def has_error(self):
        """Returns whether there was a business logic error when fetching data
        for any components for this property.

        Returns:
            boolean
        """
        return next(
            (True for cr in self.component_results
             if cr.has_error()),
            False
        )