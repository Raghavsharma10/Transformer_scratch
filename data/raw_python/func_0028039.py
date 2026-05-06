def _update_in_hdx(self, object_type, id_field_name, file_to_upload=None, **kwargs):
        # type: (str, str, Optional[str], Any) -> None
        """Helper method to check if HDX object exists in HDX and if so, update it

        Args:
            object_type (str): Description of HDX object type (for messages)
            id_field_name (str): Name of field containing HDX object identifier
            file_to_upload (Optional[str]): File to upload to HDX
            **kwargs: See below
            operation (string): Operation to perform eg. patch. Defaults to update.

        Returns:
            None
        """

        self._check_load_existing_object(object_type, id_field_name)
        # We load an existing object even thought it may well have been loaded already
        # to prevent an admittedly unlikely race condition where someone has updated
        # the object in the intervening time
        self._merge_hdx_update(object_type, id_field_name, file_to_upload, **kwargs)