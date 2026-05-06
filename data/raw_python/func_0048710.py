def get_timestamp_metadata(self):
        """Gets the metadata for a timestamp.

        return: (osid.Metadata) - metadata for the timestamp
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['timestamp'])
        metadata.update({'existing_date_time_values': self._my_map['timestamp']})
        return Metadata(**metadata)