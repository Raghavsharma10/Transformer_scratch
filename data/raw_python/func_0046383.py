def clear_start_time(self):
        """Clears the start time.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.clear_start_time_template
        if (self.get_start_time_metadata().is_read_only() or
                self.get_start_time_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['startTime'] = self._start_time_default