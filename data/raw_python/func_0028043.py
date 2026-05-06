def _delete_from_hdx(self, object_type, id_field_name):
        # type: (str, str) -> None
        """Helper method to deletes a resource from HDX

        Args:
            object_type (str): Description of HDX object type (for messages)
            id_field_name (str): Name of field containing HDX object identifier

        Returns:
            None
        """
        if id_field_name not in self.data:
            raise HDXError('No %s field (mandatory) in %s!' % (id_field_name, object_type))
        self._save_to_hdx('delete', id_field_name)