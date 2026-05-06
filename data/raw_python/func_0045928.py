def clear_provider_links(self):
        """Removes the provider chain.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.clear_assets_template
        if (self.get_provider_links_metadata().is_read_only() or
                self.get_provider_links_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['providerLinkIds'] = self._provider_links_default