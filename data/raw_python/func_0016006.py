def as_dict(self):
        """A dictionary representation of the :class:`.ObjectValidationResults`
        instance.

        Keys:
            * ``'result'``: The validation results (``True`` or ``False``)
            * ``'errors'``: A list of validation errors.
        Returns:

            A dictionary representation of an instance of this class.

        """
        d = super(ObjectValidationResults, self).as_dict()

        if self.errors:
            d['errors'] = [x.as_dict() for x in self.errors]

        return d