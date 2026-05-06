def get_upload_path(self):
        """Returns the uploaded file path from the storage backend.

        :returns: File path from the storage backend.
        :rtype: :py:class:`unicode`

        """
        location = self.get_storage().location
        return self.cleaned_data['key_name'][len(location):]