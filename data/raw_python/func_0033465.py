def fetch(self):
        """Attempt to fetch the object from the Infoblox device. If successful
        the object will be updated and the method will return True.

        :rtype: bool
        :raises: infoblox.exceptions.ProtocolError

        """
        LOGGER.debug('Fetching %s, %s', self._path, self._search_values)
        response = self._session.get(self._path, self._search_values,
                                     {'_return_fields': self._return_fields})
        if response.status_code == 200:
            values = response.json()
            self._assign(values)
            return bool(values)
        elif response.status_code >= 400:
            try:
                error = response.json()
                raise exceptions.ProtocolError(error['text'])
            except ValueError:
                raise exceptions.ProtocolError(response.content)
        return False