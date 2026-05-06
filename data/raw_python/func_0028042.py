def _create_in_hdx(self, object_type, id_field_name, name_field_name,
                       file_to_upload=None):
        # type: (str, str, str, Optional[str]) -> None
        """Helper method to check if resource exists in HDX and if so, update it, otherwise create it


        Args:
            object_type (str): Description of HDX object type (for messages)
            id_field_name (str): Name of field containing HDX object identifier
            name_field_name (str): Name of field containing HDX object name
            file_to_upload (Optional[str]): File to upload to HDX (if url not supplied)

        Returns:
            None
        """
        self.check_required_fields()
        if id_field_name in self.data and self._load_from_hdx(object_type, self.data[id_field_name]):
            logger.warning('%s exists. Updating %s' % (object_type, self.data[id_field_name]))
            self._merge_hdx_update(object_type, id_field_name, file_to_upload)
        else:
            self._save_to_hdx('create', name_field_name, file_to_upload)