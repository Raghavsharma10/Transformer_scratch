def get_created_date_metadata(self):
        """Gets the metadata for the asset creation date.

        return: (osid.Metadata) - metadata for the created date
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['created_date'])
        metadata.update({'existing_date_time_values': self._my_map['createdDate']})
        return Metadata(**metadata)