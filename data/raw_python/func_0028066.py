def delete_datastore(self):
        # type: () -> None
        """Delete a resource from the HDX datastore

        Returns:
            None
        """
        success, result = self._read_from_hdx('datastore', self.data['id'], 'resource_id',
                                              self.actions()['datastore_delete'],
                                              force=True)
        if not success:
            logger.debug(result)