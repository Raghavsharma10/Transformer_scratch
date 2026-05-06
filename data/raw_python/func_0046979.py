def clear_completion(self):
        """Clears the completion.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.grading.GradeSystemForm.clear_lowest_numeric_score
        if (self.get_completion_metadata().is_read_only() or
                self.get_completion_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['completion'] = self._completion_default