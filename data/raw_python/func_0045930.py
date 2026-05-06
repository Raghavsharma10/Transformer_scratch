def set_created_date(self, created_date):
        """Sets the created date.

        arg:    created_date (osid.calendaring.DateTime): the new
                created date
        raise:  InvalidArgument - ``created_date`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``created_date`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.set_start_time_template
        if self.get_created_date_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_date_time(
                created_date,
                self.get_created_date_metadata()):
            raise errors.InvalidArgument()
        self._my_map['createdDate'] = created_date