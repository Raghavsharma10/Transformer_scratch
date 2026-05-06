def get_allocated_time_metadata(self):
        """Gets the metadata for the allocated time.

        return: (osid.Metadata) - metadata for the allocated time
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['allocated_time'])
        metadata.update({'existing_duration_values': self._my_map['allocatedTime']})
        return Metadata(**metadata)