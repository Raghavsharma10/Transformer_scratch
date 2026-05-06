def _auto_connect(self):
        """Attempts to connect to the roaster every quarter of a second."""
        while not self._teardown.value:
            try:
                self._connect()
                return True
            except exceptions.RoasterLookupError:
                time.sleep(.25)
        return False