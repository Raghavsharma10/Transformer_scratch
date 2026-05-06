def get_end_date_metadata(self):
        """Gets the metadata for an end date.

        return: (osid.Metadata) - metadata for the date
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._end_date_metadata)
        metadata.update({'existing_date_time_values': self.my_osid_object_form._my_map['endDate']})
        return Metadata(**metadata)