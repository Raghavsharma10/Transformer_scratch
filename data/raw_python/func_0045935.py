def get_published_date_metadata(self):
        """Gets the metadata for the published date.

        return: (osid.Metadata) - metadata for the published date
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['published_date'])
        metadata.update({'existing_date_time_values': self._my_map['publishedDate']})
        return Metadata(**metadata)