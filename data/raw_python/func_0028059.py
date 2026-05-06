def update_in_hdx(self, **kwargs):
        # type: (Any) -> None
        """Check if resource exists in HDX and if so, update it

        Args:
            **kwargs: See below
            operation (string): Operation to perform eg. patch. Defaults to update.

        Returns:
            None
        """
        self._check_load_existing_object('resource', 'id')
        if self.file_to_upload and 'url' in self.data:
            del self.data['url']
        self._merge_hdx_update('resource', 'id', self.file_to_upload, **kwargs)