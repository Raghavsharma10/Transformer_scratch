def clear_duration(self):
        """Clears the duration.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.clear_duration_template
        if (self.get_duration_metadata().is_read_only() or
                self.get_duration_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['duration'] = self._duration_default