def get_start_date_metadata(self):
        """Gets the metadata for a start date.

        return: (osid.Metadata) - metadata for the date
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._start_date_metadata)
        metadata.update({'existing_date_time_values': self.my_osid_object_form._my_map['startDate']})
        return Metadata(**metadata)