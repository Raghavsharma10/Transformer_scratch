def set_start_date(self, date):
        """Sets the start date.

        arg:    date (osid.calendaring.DateTime): the new date
        raise:  InvalidArgument - ``date`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``date`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if date is None:
            raise NullArgument('date cannot be None')
        if self.get_start_date_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_date_time(date, self.get_start_date_metadata()):
            raise InvalidArgument('date must be instance of DateTime')
        self.my_osid_object_form._my_map['startDate'] = date