def clear_data(self):
        """Removes the content data.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Removes the item from filesystem and resets URL to ''
        url = self.get_url()
        # try to clear from payload first, in case that fails we won't mess with AWS
        self._payload.clear_url()
        os.remove(url)