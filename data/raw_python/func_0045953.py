def get_url_metadata(self):
        """Gets the metadata for the url.

        return: (osid.Metadata) - metadata for the url
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['url'])
        metadata.update({'existing_string_values': self._my_map['url']})
        return Metadata(**metadata)