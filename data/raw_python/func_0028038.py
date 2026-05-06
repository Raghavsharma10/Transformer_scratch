def _merge_hdx_update(self, object_type, id_field_name, file_to_upload=None, **kwargs):
        # type: (str, str, Optional[str], Any) -> None
        """Helper method to check if HDX object exists and update it

        Args:
            object_type (str): Description of HDX object type (for messages)
            id_field_name (str): Name of field containing HDX object identifier
            file_to_upload (Optional[str]): File to upload to HDX
            **kwargs: See below
            operation (string): Operation to perform eg. patch. Defaults to update.

        Returns:
            None
        """
        merge_two_dictionaries(self.data, self.old_data)
        if 'batch_mode' in kwargs:  # Whether or not CKAN should change groupings of datasets on /datasets page
            self.data['batch_mode'] = kwargs['batch_mode']
        if 'skip_validation' in kwargs:  # Whether or not CKAN should perform validation steps (checking fields present)
            self.data['skip_validation'] = kwargs['skip_validation']
        ignore_field = self.configuration['%s' % object_type].get('ignore_on_update')
        self.check_required_fields(ignore_fields=[ignore_field])
        operation = kwargs.get('operation', 'update')
        self._save_to_hdx(operation, id_field_name, file_to_upload)