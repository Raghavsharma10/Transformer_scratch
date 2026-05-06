def save(self):
        """Update the infoblox with new values for the specified object, or add
        the values if it's a new object all together.

        :raises: AssertionError
        :raises: infoblox.exceptions.ProtocolError

        """
        if 'save' not in self._supports:
            raise AssertionError('Can not save this object type')

        values = {}
        for key in [key for key in self.keys() if key not in self._save_ignore]:
            if not getattr(self, key) and getattr(self, key) != False:
                continue

            if isinstance(getattr(self, key, None), list):
                value = list()
                for item in getattr(self, key):
                    if isinstance(item, dict):
                        value.append(item)
                    elif hasattr(item, '_save_as'):
                        value.append(item._save_as())
                    elif hasattr(item, '_ref') and getattr(item, '_ref'):
                        value.append(getattr(item, '_ref'))
                    else:
                        LOGGER.warning('Cant assign %r', item)
                values[key] = value
            elif getattr(self, key, None):
                values[key] = getattr(self, key)
        if not self._ref:
            response = self._session.post(self._path, values)
        else:
            values['_ref'] = self._ref
            response = self._session.put(self._path, values)
        LOGGER.debug('Response: %r, %r', response.status_code, response.content)
        if 200 <= response.status_code <= 201:
            self.fetch()
            return True
        else:
            try:
                error = response.json()
                raise exceptions.ProtocolError(error['text'])
            except ValueError:
                raise exceptions.ProtocolError(response.content)