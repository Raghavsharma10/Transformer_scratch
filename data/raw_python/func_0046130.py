def get_provider_metadata(self):
        """Gets the metadata for a provider.

        return: (osid.Metadata) - metadata for the provider
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._provider_metadata)
        metadata.update({'existing_id_values': self.my_osid_object_form._my_map['providerId']})
        return Metadata(**metadata)