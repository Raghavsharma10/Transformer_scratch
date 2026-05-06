def clear_start_date(self):
        """Clears the start date.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_start_date_metadata().is_read_only() or
                self.get_start_date_metadata().is_required()):
            raise NoAccess()
        default_start_date = self._start_date_metadata['default_date_time_values'][0]
        self.my_osid_object_form._my_map['startDate'] = DateTime(year=default_start_date.year,
                                                                 month=default_start_date.month,
                                                                 day=default_start_date.day,
                                                                 hour=default_start_date.hour,
                                                                 minute=default_start_date.minute,
                                                                 second=default_start_date.second,
                                                                 microsecond=default_start_date.microsecond)