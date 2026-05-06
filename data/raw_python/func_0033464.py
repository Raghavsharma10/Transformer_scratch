def delete(self):
        """Remove the item from the infoblox server.

        :rtype: bool
        :raises: AssertionError
        :raises: ValueError
        :raises: infoblox.exceptions.ProtocolError

        """
        if not self._ref:
            raise ValueError('Object has no reference id for deletion')
        if 'save' not in self._supports:
            raise AssertionError('Can not save this object type')
        response = self._session.delete(self._path)
        if response.status_code == 200:
            self._ref = None
            self.clear()
            return True
        try:
            error = response.json()
            raise exceptions.ProtocolError(error['text'])
        except ValueError:
            raise exceptions.ProtocolError(response.content)