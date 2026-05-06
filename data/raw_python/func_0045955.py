def clear_url(self):
        """Removes the url.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetContentForm.clear_url_template
        if (self.get_url_metadata().is_read_only() or
                self.get_url_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['url'] = self._url_default