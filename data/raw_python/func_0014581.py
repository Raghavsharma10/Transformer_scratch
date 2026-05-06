def delete(self, handle):
        """Delete the specified object.

        Arguments:
        handle -- Handle of object to delete.

        """
        self._check_session()
        self._rest.delete_request('objects', str(handle))