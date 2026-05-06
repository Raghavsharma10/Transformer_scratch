def has_datastore(self):
        # type: () -> bool
        """Check if the resource has a datastore.

        Returns:
            bool: Whether the resource has a datastore or not
        """
        success, result = self._read_from_hdx('datastore', self.data['id'], 'resource_id',
                                              self.actions()['datastore_search'])
        if not success:
            logger.debug(result)
        else:
            if result:
                return True
        return False