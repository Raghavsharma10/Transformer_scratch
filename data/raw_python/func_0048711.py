def set_timestamp(self, timestamp):
        """Sets the timestamp.

        arg:    timestamp (osid.calendaring.DateTime): the new timestamp
        raise:  InvalidArgument - ``timestamp`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``timestamp`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.set_start_time_template
        if self.get_timestamp_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_date_time(
                timestamp,
                self.get_timestamp_metadata()):
            raise errors.InvalidArgument()
        self._my_map['timestamp'] = timestamp