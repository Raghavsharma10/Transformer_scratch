def _check_load_existing_object(self, object_type, id_field_name, operation='update'):
        # type: (str, str, str) -> None
        """Check metadata exists and contains HDX object identifier, and if so load HDX object

        Args:
            object_type (str): Description of HDX object type (for messages)
            id_field_name (str): Name of field containing HDX object identifier
            operation (str): Operation to report if error. Defaults to update.

        Returns:
            None
        """
        self._check_existing_object(object_type, id_field_name)
        if not self._load_from_hdx(object_type, self.data[id_field_name]):
            raise HDXError('No existing %s to %s!' % (object_type, operation))