def set_allocated_time(self, time):
        """Sets the allocated time.

        arg:    time (osid.calendaring.Duration): the allocated time
        raise:  InvalidArgument - ``time`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.set_duration_template
        if self.get_allocated_time_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_duration(
                time,
                self.get_allocated_time_metadata()):
            raise errors.InvalidArgument()
        map = dict()
        map['days'] = time.days
        map['seconds'] = time.seconds
        map['microseconds'] = time.microseconds
        self._my_map['allocatedTime'] = map