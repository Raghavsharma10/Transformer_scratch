def set_url(self, url):
        """Sets the url.

        arg:    url (string): the new copyright
        raise:  InvalidArgument - ``url`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``url`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetContentForm.set_url_template
        if self.get_url_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_string(
                url,
                self.get_url_metadata()):
            raise errors.InvalidArgument()
        self._my_map['url'] = url