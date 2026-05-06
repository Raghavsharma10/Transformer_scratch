def set_completion(self, completion):
        """Sets the completion percentage.

        arg:    completion (decimal): the completion percentage
        raise:  InvalidArgument - ``completion`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.grading.GradeSystemForm.set_lowest_numeric_score
        if self.get_completion_metadata().is_read_only():
            raise errors.NoAccess()
        try:
            completion = float(completion)
        except ValueError:
            raise errors.InvalidArgument()
        if not self._is_valid_decimal(completion, self.get_completion_metadata()):
            raise errors.InvalidArgument()
        self._my_map['completion'] = completion