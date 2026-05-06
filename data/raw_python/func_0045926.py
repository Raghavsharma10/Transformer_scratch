def get_provider_links_metadata(self):
        """Gets the metadata for the provider chain.

        return: (osid.Metadata) - metadata for the provider chain
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.get_assets_metadata_template
        metadata = dict(self._mdata['provider_links'])
        metadata.update({'existing_provider_links_values': self._my_map['providerLinkIds']})
        return Metadata(**metadata)