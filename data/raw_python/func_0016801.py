def first(self):
        """Return the first result.  If there are no results, raises :exc:`~bloop.exceptions.ConstraintViolation`.

        :return: The first result.
        :raises bloop.exceptions.ConstraintViolation: No results.
        """
        self.reset()
        value = next(self, None)
        if value is None:
            raise ConstraintViolation("{} did not find any results.".format(self.mode.capitalize()))
        return value