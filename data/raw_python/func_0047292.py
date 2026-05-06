def clear_allocated_time(self):
        """Clears the allocated time.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.clear_duration_template
        if (self.get_allocated_time_metadata().is_read_only() or
                self.get_allocated_time_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['allocatedTime'] = self._allocated_time_default