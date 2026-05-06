def get_processed_path(self):
        """Returns the processed file path from the storage backend.

        :returns: File path from the storage backend.
        :rtype: :py:class:`unicode`

        """
        location = self.get_storage().location
        return self.get_processed_key_name()[len(location):]