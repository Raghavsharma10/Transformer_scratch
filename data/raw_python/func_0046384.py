def get_deadline_metadata(self):
        """Gets the metadata for the assessment deadline.

        return: (osid.Metadata) - metadata for the end time
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['deadline'])
        metadata.update({'existing_date_time_values': self._my_map['deadline']})
        return Metadata(**metadata)