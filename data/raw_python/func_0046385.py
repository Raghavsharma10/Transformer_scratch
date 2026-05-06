def set_deadline(self, end):
        """Sets the assessment end time.

        arg:    end (timestamp): assessment end time
        raise:  InvalidArgument - ``end`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.set_start_time_template
        if self.get_deadline_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_timestamp(
                end,
                self.get_deadline_metadata()):
            raise errors.InvalidArgument()
        self._my_map['deadline'] = end