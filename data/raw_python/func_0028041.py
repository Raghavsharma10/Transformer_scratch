def _save_to_hdx(self, action, id_field_name, file_to_upload=None):
        # type: (str, str, Optional[str]) -> None
        """Creates or updates an HDX object in HDX, saving current data and replacing with returned HDX object data
        from HDX

        Args:
            action (str): Action to perform: 'create' or 'update'
            id_field_name (str): Name of field containing HDX object identifier
            file_to_upload (Optional[str]): File to upload to HDX

        Returns:
            None
        """
        result = self._write_to_hdx(action, self.data, id_field_name, file_to_upload)
        self.old_data = self.data
        self.data = result