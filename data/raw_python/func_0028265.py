def has_object_error(self):
        """Returns true if any requested object had a business logic error,
        otherwise returns false

        Returns:
            boolean
        """
        if self._has_object_error is None:
            # scan the objects for any business error codes
            self._has_object_error = next(
                (True for o in self.objects()
                 if o.has_error()),
                False)
        return self._has_object_error