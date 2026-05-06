def clear_end_date(self):
        """Clears the end date.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_end_date_metadata().is_read_only() or
                self.get_end_date_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['endDate'] = \
            DateTime(**self._end_date_metadata['default_date_time_values'][0])