def create_in_hdx(self):
        # type: () -> None
        """Check if resource exists in HDX and if so, update it, otherwise create it

        Returns:
            None
        """
        self.check_required_fields()
        id = self.data.get('id')
        if id and self._load_from_hdx('resource', id):
            logger.warning('%s exists. Updating %s' % ('resource', id))
            if self.file_to_upload and 'url' in self.data:
                del self.data['url']
            self._merge_hdx_update('resource', 'id', self.file_to_upload)
        else:
            self._save_to_hdx('create', 'name', self.file_to_upload)