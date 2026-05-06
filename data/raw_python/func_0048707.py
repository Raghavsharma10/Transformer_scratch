def get_priority_metadata(self):
        """Gets the metadata for a priority type.

        return: (osid.Metadata) - metadata for the priority
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.logging.LogEntryForm.get_priority_metadata
        metadata = dict(self._mdata['priority'])
        metadata.update({'existing_type_values': self._my_map['priorityId']})
        return Metadata(**metadata)