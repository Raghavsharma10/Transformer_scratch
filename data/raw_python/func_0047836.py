def is_authenticated(self):
        """Tests if an agent is authenticated to this session.

        return: (boolean) - true if valid authentication credentials
                exist, false otherwise
        compliance: mandatory - This method must be implemented.

        """
        if self._proxy is None:
            return False
        elif self._proxy.has_authentication():
            return self._proxy.get_authentication().is_valid()
        else:
            return False