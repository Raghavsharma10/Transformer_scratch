def set_duration(self, duration):
        """Sets the assessment duration.

        arg:    duration (osid.calendaring.Duration): assessment
                duration
        raise:  InvalidArgument - ``duration`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.set_duration_template
        if self.get_duration_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_duration(
                duration,
                self.get_duration_metadata()):
            raise errors.InvalidArgument()
        map = dict()
        map['days'] = duration.days
        map['seconds'] = duration.seconds
        map['microseconds'] = duration.microseconds
        self._my_map['duration'] = map