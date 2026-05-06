def clear_deadline(self):
        """Clears the deadline.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.clear_start_time_template
        if (self.get_deadline_metadata().is_read_only() or
                self.get_deadline_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['deadline'] = self._deadline_default