def clear_assets(self):
        """Clears the assets.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.clear_assets_template
        if (self.get_assets_metadata().is_read_only() or
                self.get_assets_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['assetIds'] = self._assets_default