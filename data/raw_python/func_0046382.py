def set_start_time(self, start):
        """Sets the assessment start time.

        arg:    start (osid.calendaring.DateTime): assessment start time
        raise:  InvalidArgument - ``start`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.set_start_time_template
        if self.get_start_time_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_date_time(
                start,
                self.get_start_time_metadata()):
            raise errors.InvalidArgument()
        self._my_map['startTime'] = start