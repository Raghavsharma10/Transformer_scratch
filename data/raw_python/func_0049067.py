def get_start_date_metadata(self):
        """Gets the metadata for a start date.

        return: (osid.Metadata) - metadata for the date
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._mdata['start_date'])
        metadata.update({'existing_date_time_values': self._my_map['startDate']})
        return Metadata(**metadata)