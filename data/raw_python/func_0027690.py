def namespace(self):
        """
        Return a dictionary representing the namespace which should be
        available to the user.
        """
        self._ns = {
            'db': self.store,
            'store': store,
            'autocommit': False,
            }
        return self._ns