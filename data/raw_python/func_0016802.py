def one(self):
        """Return the unique result.  If there is not exactly one result,
        raises :exc:`~bloop.exceptions.ConstraintViolation`.

        :return: The unique result.
        :raises bloop.exceptions.ConstraintViolation: Not exactly one result.
        """
        first = self.first()
        second = next(self, None)
        if second is not None:
            raise ConstraintViolation("{} found more than one result.".format(self.mode.capitalize()))
        return first