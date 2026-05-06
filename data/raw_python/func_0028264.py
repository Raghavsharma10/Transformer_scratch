def get_object_errors(self):
        """Gets a list of business error message strings
        for each of the requested objects that had a business error.
        If there was no error, returns an empty list

        Returns:
            List of strings
        """
        if self._object_errors is None:
            self._object_errors = [{str(o): o.get_errors()}
                                   for o in self.objects()
                                   if o.has_error()]

        return self._object_errors